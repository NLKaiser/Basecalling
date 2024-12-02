import numpy as np
import tensorflow as tf

import re
import parasail
from collections import defaultdict

from Bio import pairwise2

"""
Contains utility functions
and
the implementation to 
"""

split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")

class CosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, final_lr, decay_steps):
        super(CosineDecaySchedule, self).__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay_steps = decay_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)
        cosine_decay = 0.5 * (tf.cos(np.pi * step / decay_steps) + 1.0)
        decayed_lr = (self.initial_lr - self.final_lr) * cosine_decay + self.final_lr
        return decayed_lr

def get_learning_rate(initial_lr=1e-4, final_lr=1e-6, decay_steps=100, epochs=None, steps_per_epoch=None):
    if (epochs != None and steps_per_epoch != None):
        decay_steps = epochs * steps_per_epoch
    return CosineDecaySchedule(initial_lr, final_lr, decay_steps)
    
class LinearDecayScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, final_learning_rate, total_epochs, steps_per_epoch):
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.total_steps = total_epochs * steps_per_epoch  # Total number of steps across all epochs

    def __call__(self, step):
        # Cast the current step as float
        step = tf.cast(step, tf.float32)

        # Linear decay formula based on the step
        decayed_learning_rate = self.initial_learning_rate - \
            (step / self.total_steps) * (self.initial_learning_rate - self.final_learning_rate)

        return decayed_learning_rate

def decode_ref(encoded, labels):
    """
    Convert a integer encoded reference into a string and remove blanks
    """
    return ''.join(labels[e] for e in encoded.tolist() if e)

def parasail_to_sam(result, seq):
    """
    Extract reference start and sam compatible cigar string.

    :param result: parasail alignment result.
    :param seq: query sequence.

    :returns: reference start coordinate, cigar string.
    """
    cigstr = result.cigar.decode.decode()
    first = re.search(split_cigar, cigstr)

    first_count, first_op = first.groups()
    prefix = first.group()
    rstart = result.cigar.beg_ref
    cliplen = result.cigar.beg_query

    clip = '' if cliplen == 0 else '{}S'.format(cliplen)
    if first_op == 'I':
        pre = '{}S'.format(int(first_count) + cliplen)
    elif first_op == 'D':
        pre = clip
        rstart = int(first_count)
    else:
        pre = '{}{}'.format(clip, prefix)

    mid = cigstr[len(prefix):]
    end_clip = len(seq) - result.end_query - 1
    suf = '{}S'.format(end_clip) if end_clip > 0 else ''
    new_cigstr = ''.join((pre, mid, suf))
    return rstart, new_cigstr

def accuracy(ref, seq, balanced=False, min_coverage=0.0):
    """
    Calculate the accuracy between `ref` and `seq`
    """
    alignment = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
    counts = defaultdict(int)

    q_coverage = len(alignment.traceback.query) / len(seq)
    r_coverage = len(alignment.traceback.ref) / len(ref)

    if r_coverage < min_coverage:
        return 0.0

    _, cigar = parasail_to_sam(alignment, seq)

    for count, op  in re.findall(split_cigar, cigar):
        counts[op] += int(count)

    if balanced:
        accuracy = (counts['='] - counts['I']) / (counts['='] + counts['X'] + counts['D'])
    else:
        accuracy = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])
    return accuracy * 100

def accuracy_global(ref, seq, balanced=False, min_coverage=0.0):
    """
    Calculate the accuracy between `ref` and `seq`
    """
    alignment = parasail.nw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
    counts = defaultdict(int)

    q_coverage = len(alignment.traceback.query) / len(seq)
    r_coverage = len(alignment.traceback.ref) / len(ref)

    if r_coverage < min_coverage:
        return 0.0

    _, cigar = parasail_to_sam(alignment, seq)

    for count, op  in re.findall(split_cigar, cigar):
        counts[op] += int(count)

    if balanced:
        accuracy = (counts['='] - counts['I']) / (counts['='] + counts['X'] + counts['D'])
    else:
        accuracy = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])
    return accuracy * 100

def accuracy_pairwise(ref, seq, min_coverage=0.0):
    alignment = pairwise2.align.globalxx(seq, ref)
    score = alignment[0].score
    if score < min_coverage * len(ref):
        return 0.0
    
    return score/len(alignment[0].seqB)

def alignment_local(seq, ref):
    return parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)

def alignment_global(seq, ref):
    return parasail.nw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)

def model_summary_to_string(model):
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return "\n".join(summary_lines)

