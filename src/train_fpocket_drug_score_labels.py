import sys
import os
import random
import math
from datetime import datetime
from glob import glob
import yaml


# --- INPUTS ---
# --- LOAD CONFIG FROM YAML ---
yaml_filename = sys.argv[1]
with open(yaml_filename, "r") as stream:
    try:
        training_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# DATA_DIR: /Path/to/(ae-)pocketminer/data/ 
# PRECOMPUTED_DIR: Only needed if use_tensors or use_lm is True
DATA_DIR = training_config.get('data_dir', "../data/")
PRECOMPUTED_DIR = training_config.get('precomputed_dir', "../data/")

os.environ['POCKETMINER_DATA_DIR'] = DATA_DIR
# '' -> original PocketMiner training data; set e.g. '_plus_36prots_fah_rmVP35'
# to use the larger training set instead. See datasets.py for the full list
# of available suffixes.
os.environ['TRAIN_DATA_SUFFIX'] = training_config.get('train_data_suffix', '')
# --- INPUTS ---    

# tf.keras is Keras 2 (tf-keras), like the models in ae_pocketminer; this must
# be set before TensorFlow is first imported
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import tensorflow as tf
import tqdm
import mdtraj as md
import numpy as np
from tensorflow import keras as keras

from datasets import *
from ae_pocketminer.models import *
from util import save_checkpoint, load_checkpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# utility method for splitting lists
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def extract_epoch(file_path):
    # Get the file name from the path
    file_name = file_path.split('/')[-1]
    # Extract the number after the first underscore and before ".index"
    epoch_part = file_name.split('_')[1]
    return int(epoch_part.split('.')[0])

# utility method for continuing trainings
def get_training_data_for_restart(model, opt, nn_dir):
    # Determine network name
    file_paths = glob(f"{nn_dir}/*.index")
    index_filenames = sorted(file_paths, key=extract_epoch)

    print("index_filenames:", index_filenames, flush=True)
    # check if there is only one network in the folder
    if len(np.unique([os.path.basename(fn).split('_')[0] for fn in index_filenames])) == 1:
        nn_path = index_filenames[-1].split('.index')[0]
        print("nn_path:", nn_path, flush=True)
        print("Loading checkpoint...", flush=True)
        load_checkpoint(model, opt, nn_path)
    else:
        print('multiple networks in folder')
        nn_path = index_filenames[-1].split('.index')[0]
        load_checkpoint(model, opt, nn_path)
        print(f'selecting following network id: {os.path.basename(nn_path)}')

    model_id = os.path.basename(index_filenames[-1]).split('_')[0]
    last_epoch = int(os.path.basename(index_filenames[-1]).split('_')[1].split('.')[0])

    # need to also determine best epoch, best val, and best auc
    best_epoch, best_val, best_pr_auc = 0, np.inf, 0
    val_losses = []
    train_losses = []
    for epoch in range(last_epoch + 1):
        train_loss = np.load(os.path.join(nn_dir, f"train_loss_{epoch}.npy"))
        train_losses.append(train_loss)
        val_loss = np.load(os.path.join(nn_dir, f"val_loss_{epoch}.npy"))
        val_losses.append(val_loss)
        pr_auc = np.load(os.path.join(nn_dir, f"val_pr_auc_{epoch}.npy"))
        if val_loss < best_val:
            best_val = val_loss
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_epoch = epoch

    print('model_id', 'last_epoch', 'best_epoch', 'best_val', 'best_pr_auc', flush=True)
    print(model_id, last_epoch, best_epoch, best_val, best_pr_auc, flush=True)
    return model_id, last_epoch, best_epoch, best_val, best_pr_auc, val_losses, train_losses


def make_model():
    # use_attention/attention_heads now explicit; must match whichever
    # architecture `previous_nn_dir` was trained
    # with -- see use_attention default note in the yaml files below.
    model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                     hidden_dim=(16, HIDDEN_DIM),
                     attention_heads=NUM_HEADS,
                     num_layers=NUM_LAYERS, dropout=DROPOUT_RATE,
                     ablate_sidechain_vectors=ablate_sidechain_vectors,
                     use_attention=use_attention)
    return model


def main():
    # Prepare data
    print(f'\nusing {FILESTEM} training data')
    # batch size = N proteins
    trainset = simulation_dataset(BATCH_SIZE, FILESTEM,
                                  use_tensors=use_tensors,
                                  y_type='float32')
    print('training data loaded')

    # Set optimizer and make model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model = make_model()

    if residue_batches:
        train_func = train_residue_batches
    else:
        train_func = train_protein_batches

    if continue_previous_training:
        training_params = get_training_data_for_restart(model, optimizer, previous_nn_dir)
        model_id, last_epoch, best_epoch, best_val, best_pr_auc, val_losses, train_losses = training_params
        start_epoch = 0
    else:
        model_id = int(datetime.timestamp(datetime.now()))
        best_epoch, best_val, best_pr_auc = 0, np.inf, 0
        val_losses = []
        train_losses = []
        start_epoch = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        loss, y_pred, y_true = train_func(trainset, model, optimizer=optimizer)

        train_losses.append(loss)
        print('\nEPOCH {} training loss: {}'.format(epoch, loss))

        # For debugging save out train predictions and true values
        np.save(os.path.join(outdir, f"train_loss_{epoch}.npy"), loss)
        np.save(os.path.join(outdir, f"train_y_pred_{epoch}.npy"), np.array(y_pred, dtype=object))
        np.save(os.path.join(outdir, f"train_y_true_{epoch}.npy"),  np.array(y_true, dtype=object))

        save_checkpoint(model_path, model, optimizer, model_id, epoch)

        # Determine validation loss and performance statistics
        predictions, mask = predict_on_xtals(model, xtal_val_ids, use_tensors=use_tensors)
        np.save(os.path.join(outdir, f"val_predictions_{epoch}.npy"), predictions)

        loss, auc, pr_auc, y_pred, y_true = assess_performance(predictions, mask, val_label_dicts)

        # Save out validation metrics for this epoch
        np.save(os.path.join(outdir, f"val_loss_{epoch}.npy"), loss)
        np.save(os.path.join(outdir, f"val_auc_{epoch}.npy"), auc)
        np.save(os.path.join(outdir, f"val_pr_auc_{epoch}.npy"), pr_auc)
        np.save(os.path.join(outdir, f"val_y_pred_{epoch}.npy"), y_pred)
        np.save(os.path.join(outdir, f"val_y_true_{epoch}.npy"), y_true)

        val_losses.append(loss)
        print('\nEPOCH {} validation loss: {}'.format(epoch, loss))
        if loss < best_val:
            best_val = loss

        # Update best PR AUC to keep track of best model
        if pr_auc > best_pr_auc:
            best_epoch, best_pr_auc = epoch, pr_auc

    # Test with best validation loss
    print(f'\nBest AUC is in epoch {best_epoch}')
    path = model_path.format(str(model_id).zfill(3), str(best_epoch).zfill(3))

    # Save out training and validation losses
    np.save(f'{outdir}/cv_loss.npy', val_losses)
    np.save(f'{outdir}/train_loss.npy', train_losses)

    load_checkpoint(model, optimizer, path)

    predictions, mask = predict_on_xtals(model, test_ids, use_tensors=use_tensors)
    np.save(os.path.join(outdir, f"test_predictions.npy"), predictions)

    loss, tp, fp, tn, fn, acc, prec, recall, auc, pr_auc, y_pred, y_true = assess_performance(predictions,
        mask, test_label_dicts, test=True)
    print('\nEPOCH TEST {:.4f} {:.4f}'.format(loss, acc))

    return loss, tp, fp, tn, fn, acc, prec, recall, auc, pr_auc, y_pred, y_true


def assess_performance(predictions, mask, label_dicts, test=False):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc
    from sklearn.metrics import roc_auc_score

    auc_metric.reset_states()
    pr_auc_metric.reset_states()

    label_dictionary = np.load(label_dicts, allow_pickle=True).item()

    l_max = max([len(l) for l in label_dictionary.values()])

    label_mask = np.zeros([len(label_dictionary.keys()), l_max], dtype=bool)
    for i, l in enumerate(label_dictionary.values()):
        label_mask[i] = np.pad(l != 2, [[0, l_max - len(l)]])

    true_labels = np.zeros([len(label_dictionary.keys()), l_max], dtype=int)
    for i, l in enumerate(label_dictionary.values()):
        true_labels[i] = np.pad(l, [[0, l_max - len(l)]])

    if test:
        print('\nBalance in test set:')
    else:
        print('\nBalance in val set:')
    print(f'Total negative residues: {np.sum(true_labels[label_mask] == 0)}')
    print(f'Total positive residues: {np.sum(true_labels[label_mask] == 1)}')

    y_true = true_labels[label_mask]
    y_pred = predictions[mask.astype(bool) & label_mask]
    loss = loss_fn(y_true, y_pred)

    if test:
        tp_metric.update_state(y_true, y_pred)
        fp_metric.update_state(y_true, y_pred)
        tn_metric.update_state(y_true, y_pred)
        fn_metric.update_state(y_true, y_pred)
        acc_metric.update_state(y_true, y_pred)
        prec_metric.update_state(y_true, y_pred)
        recall_metric.update_state(y_true, y_pred)
        auc_metric.update_state(y_true, y_pred)
        pr_auc_metric.update_state(y_true, y_pred)
    else:
        auc_metric.update_state(y_true, y_pred)
        pr_auc_metric.update_state(y_true, y_pred)

    if test:
        tp = tp_metric.result().numpy()
        fp = fp_metric.result().numpy()
        tn = tn_metric.result().numpy()
        fn = fn_metric.result().numpy()
        acc = acc_metric.result().numpy()
        prec = prec_metric.result().numpy()
        recall = recall_metric.result().numpy()
        auc = auc_metric.result().numpy()
        pr_auc = pr_auc_metric.result().numpy()
    else:
        auc = auc_metric.result().numpy()
        pr_auc = pr_auc_metric.result().numpy()

    if test:
        return loss, tp, fp, tn, fn, acc, prec, recall, auc, pr_auc, y_pred, y_true
    else:
        return loss, auc, pr_auc, y_pred, y_true


def train_residue_batches(dataset, model, optimizer=None,
                          positive_weight=1, negative_weight=1):
    losses = []
    y_pred, y_true = [], []

    for batch in dataset:
        X, S, y, meta, M = batch
        iis = get_indices(y)

        num_batches = int(math.ceil(len(iis) / NUMBER_RESIDUES_PER_BATCH))
        iis_split = list(split(iis, num_batches))
        for i, iis in enumerate(iis_split):
            with tf.GradientTape() as tape:
                prediction = model(X, S, M, train=True, res_level=True)
                y_sel = tf.gather_nd(y, indices=iis)
                assert np.all(y_sel >= 0)
                prediction = tf.gather_nd(prediction, indices=iis)
                loss_value = loss_fn(y_sel, prediction)

            assert(np.isfinite(float(loss_value)))
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            losses.append(float(loss_value))
            y_pred.extend(prediction.numpy().tolist())
            y_true.extend(y_sel.numpy().tolist())

    return np.mean(losses), y_pred, y_true


def train_protein_batches(dataset, model, optimizer=None, positive_weight=1, negative_weight=1):
    losses = []
    y_pred, y_true = [], []

    for batch in dataset:
        X, S, y, meta, M = batch
        with tf.GradientTape() as tape:
            prediction = model(X, S, M, train=True, res_level=True)
            loss_value = loss_fn(y, prediction)

        assert(np.isfinite(float(loss_value)))
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        losses.append(float(loss_value))
        y_pred.extend(prediction.numpy().tolist())
        y_true.extend(y.numpy().tolist())

    return np.mean(losses), y_pred, y_true


def get_indices(y):
    iis = [[struct_index, res_index]
           for struct_index, y_vals in enumerate(y)
           for res_index in np.where(y_vals >= 0)[0]]
    return iis


def process_paths(apo_IDs, use_tensors=True):
    """Takes a list of apo IDs to pdb files
    """
    pdb_dir = os.path.join(DATA_DIR, 'pm-dataset/all-structures/')
    paths = [pdb_dir + e + '_clean_h.pdb' for e in apo_IDs]

    paths = [p if os.path.exists(p)
             else pdb_dir + apo_IDs[i].upper() + '_clean_h.pdb'
             for i, p in enumerate(paths)]

    strucs = [md.load(p) for p in paths]
    pdbs = []
    for s in strucs:
        prot_iis = s.top.select("protein and (name N or name CA or name C or name O)")
        prot_bb = s.atom_slice(prot_iis)
        pdbs.append(prot_bb)

    B = len(strucs)
    L_max = np.max([p.top.n_residues for p in pdbs])

    if use_tensors:
        X = np.zeros([B, L_max, 5, 3], dtype=np.float32)
    else:
        X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    S = np.zeros([B, L_max], dtype=np.int32)

    for i, prot_bb in enumerate(pdbs):
        l = prot_bb.top.n_residues

        if use_tensors:
            fn = f'X_tensor_{apo_IDs[i]}.npy'
            X_path = os.path.join(PRECOMPUTED_DIR, f'precompute_X/{fn}')
            if os.path.exists(X_path):
                xyz = np.load(X_path)
            else:
                # try capitalizing the apo PDB ID
                fn = f'X_tensor_{apo_IDs[i].upper()}.npy'
                X_path = os.path.join(PRECOMPUTED_DIR, f'precompute_X/{fn}')
                xyz = np.load(X_path)
        else:
            xyz = prot_bb.xyz.reshape(l, 4, 3)

        seq = [r.name for r in prot_bb.top.residues]
        S[i, :l] = np.asarray([lookup[abbrev[a]] for a in seq], dtype=np.int32)
        X[i] = np.pad(xyz, [[0,L_max-l], [0,0], [0,0]],
                      'constant', constant_values=(np.nan, ))

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    X = np.nan_to_num(X)

    return X, S, mask


def predict_on_xtals(model, xtal_set, use_tensors=True):

    val_set_apo_ids = np.load(xtal_set, allow_pickle=True)
    X, S, mask = process_paths(val_set_apo_ids, use_tensors=use_tensors)

    prediction = model(X, S, mask, train=False, res_level=True)
    mask = mask.astype(bool)
    return prediction, mask


######### INPUTS ##########
## Define global variables
# yaml_filename / training_config were already loaded once at the top

# ------TRAINING PARAMETERS----- #
NUM_EPOCHS = training_config['NUM_EPOCHS']
# BATCH_SIZE specifies the number of proteins that are
# featurized and drawn each iteration of the training loop
BATCH_SIZE = training_config['BATCH_SIZE']

residue_batches = training_config['residue_batches']
if residue_batches:
    # NUMBER_RESIDUES_PER_BATCH specifies the number of residues
    # that are used for an error calculation
    NUMBER_RESIDUES_PER_BATCH = training_config['NUMBER_RESIDUES_PER_BATCH']

LEARNING_RATE = training_config['LEARNING_RATE']

continue_previous_training = training_config['continue_previous_training']
if continue_previous_training:
    previous_nn_dir = training_config['previous_nn_dir']

# ----------------------------- #

use_tensors = training_config['use_tensors']

# ----- FEATURIZATION PARAMS ---- #

feat_type = training_config['feat_type']
stride = training_config['stride']
window = training_config['window']
cutoff = training_config['cutoff']

# ---END FEATURIZATION PARAMS ---- #

# ------- GVP INPUT PARAMETERS ---- #
DROPOUT_RATE = training_config['DROPOUT_RATE']
HIDDEN_DIM = training_config['HIDDEN_DIM']
NUM_LAYERS = training_config['NUM_LAYERS']

NUM_HEADS = training_config['NUM_HEADS']

if 'ablate_sidechain_vectors' in training_config:
    ablate_sidechain_vectors = training_config['ablate_sidechain_vectors']
else:
    ablate_sidechain_vectors = True

# --- ATTENTION ---
# use_attention=True -> AE-PocketMiner (NEW)
# use_attention=False -> original PocketMiner architecture
# This MUST match how `previous_nn_dir` was trained, or load_checkpoint()
# in get_training_data_for_restart() will fail/mismatch.
if 'use_attention' in training_config:
    use_attention = training_config['use_attention']
else:
    use_attention = True

# -----END GVP INPUT PARAMETERS ---- #

# ---- XTAL VALIDATION SET -----#
# file contains pdb path as well as labels
# xtal_validation_path = training_config['xtal_validation_path']

xtal_val_ids = training_config['xtal_val_ids']
val_label_dicts = training_config['val_label_dicts']

test_ids = training_config['test_ids']
test_label_dicts = training_config['test_label_dicts']
# ---- END XTAL VALIDATION SET -----#

# result dir
base_path = training_config['base_path']
######### END INPUTS ##############


####### CREATE OUTPUT FILENAMES #####
subdir_name = 'fpocket-drug-scores-unbinarized'

if residue_batches:
    subdir_name += f'-train-with-{NUMBER_RESIDUES_PER_BATCH}-residue-batches'
    batch_string = f"b{NUMBER_RESIDUES_PER_BATCH}resis_b{BATCH_SIZE}proteins"
else:
    subdir_name += f'-train-with-{BATCH_SIZE}-protein-batches'
    batch_string = f"b{BATCH_SIZE}proteins"


nn_name = (f"net_8-50_1-32_16-100_"
           f"dr_{DROPOUT_RATE}_"
           f"nl_{NUM_LAYERS}_hd_{HIDDEN_DIM}_"
           f"lr_{LEARNING_RATE}_{batch_string}_"
           f"{NUM_EPOCHS}epoch_"
           f"feat_method_fpocket_drug_scores_"
           f"{feat_type}_window_{window}_cutoff_{cutoff}_stride_{stride}")

if not ablate_sidechain_vectors:
    nn_name += "_sidechains"

outdir = f'{base_path}/{subdir_name}/{nn_name}/'

####### END CREATE OUTPUT FILENAMES #####

if continue_previous_training:
    outdir = (previous_nn_dir +
              f"_refine_feat_method_fpocket_drug_scores_"
              f"{feat_type}_window_{window}_cutoff_{cutoff}_stride_{stride}/")

print(outdir)

os.makedirs(outdir, exist_ok=True)
model_path = outdir + '{}_{}'
FILESTEM = f'fpocket-drug-scores-{feat_type}-cutoff-{cutoff}-window-{window}ns-stride-{stride}'
print(FILESTEM)

#### GPU INFO ####
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)
#####################

##### PERFORMANCE METRICS ########
tp_metric = keras.metrics.TruePositives(name='tp')
fp_metric = keras.metrics.FalsePositives(name='fp')
tn_metric = keras.metrics.TrueNegatives(name='tn')
fn_metric = keras.metrics.FalseNegatives(name='fn')
acc_metric = keras.metrics.BinaryAccuracy(name='accuracy')
prec_metric = keras.metrics.Precision(name='precision')
recall_metric = keras.metrics.Recall(name='recall')
auc_metric = keras.metrics.AUC(name='auc')
pr_auc_metric = keras.metrics.AUC(curve='PR', name='pr_auc')

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

##################################

######## RUN TRAINING ############
loss, tp, fp, tn, fn, acc, prec, recall, auc, pr_auc, y_pred, y_true = main()
#################################

####### SAVE TEST RESULTS ########
np.save(os.path.join(outdir,"test_loss.npy"), loss)
np.save(os.path.join(outdir,"test_tp.npy"), tp)
np.save(os.path.join(outdir,"test_fp.npy"), fp)
np.save(os.path.join(outdir,"test_tn.npy"), tn)
np.save(os.path.join(outdir,"test_fn.npy"), fn)
np.save(os.path.join(outdir,"test_acc.npy"), acc)
np.save(os.path.join(outdir,"test_prec.npy"), prec)
np.save(os.path.join(outdir,"test_recall.npy"), recall)
np.save(os.path.join(outdir,"test_auc.npy"), auc)
np.save(os.path.join(outdir,"test_pr_auc.npy"), pr_auc)
np.save(os.path.join(outdir,"test_y_pred.npy"), y_pred)
np.save(os.path.join(outdir,"test_y_true.npy"), y_true)
##################################

