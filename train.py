from __future__ import division
import numpy as np
import torch
import logging
import soundfile as sf
import time
import six
import argparse
import threading
import os
import Queue
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
from wavenet import WaveNet, initialize, encode_mu_law
from utils import find_files, read_hdf5
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta


def validate_length(x, y):
    """VALIDATE LENGTH"""
    if x.shape[0] < y.shape[0]:
        y = y[:x.shape[0]]
    if x.shape[0] > y.shape[0]:
        x = x[:y.shape[0]]
    assert len(x) == len(y)

    return x, y


def custom_generator(wav_dir, feat_dir, receptive_field=None, batch_size=None,
                     wav_transform=None, feat_transform=None, shuffle=True,
                     use_speaker_code=False):
    """TRAINING BATCH GENERATOR"""
    # get file list
    filenames = sorted(find_files(wav_dir, "*.wav", use_dir_name=False))
    wav_list = [wav_dir + "/" + filename for filename in filenames]
    feat_list = [feat_dir + "/" + filename.replace(".wav", ".h5") for filename in filenames]

    # shuffle list
    if shuffle:
        n_files = len(wav_list)
        idx = np.random.permutation(n_files)
        wav_list = [wav_list[i] for i in idx]
        feat_list = [feat_list[i] for i in idx]

    # generator part
    while True:
        for wavfile, featfile in zip(wav_list, feat_list):
            x, fs = sf.read(wavfile, dtype=np.float32)
            h = read_hdf5(featfile, "/feat")
            if use_speaker_code:
                sc = read_hdf5(featfile, "/speaker_code")
                sc = np.tile(sc, [h.shape[0], 1])
                h = np.concatenate([h, sc], axis=1)

            # check both lengths are same
            x, h = validate_length(x, h)

            # cut utterance into small batch
            if batch_size is not None:
                x_buffer = np.empty((0), dtype=np.float32)
                h_buffer = np.empty((0, h.shape[1]), dtype=np.float32)
                x_buffer = np.concatenate([x_buffer, x], axis=0)
                h_buffer = np.concatenate([h_buffer, h], axis=0)

                while len(x_buffer) > receptive_field + batch_size:
                    x_ = x_buffer[:receptive_field + batch_size]
                    h_ = h_buffer[:receptive_field + batch_size]

                    if wav_transform is not None:
                        x_ = wav_transform(x_)
                    if feat_transform is not None:
                        h_ = feat_transform(h_)

                    batch_x = x_[:-1].unsqueeze(0)
                    batch_h = h_[:-1].transpose(0, 1).unsqueeze(0)
                    batch_target = x_[1:]

                    yield (batch_x, batch_h), batch_target

                    x_buffer = x_buffer[batch_size:]
                    h_buffer = h_buffer[batch_size:]

            # utterance batch
            else:
                if wav_transform is not None:
                    x = wav_transform(x)
                if feat_transform is not None:
                    h = feat_transform(h)

                batch_x = x[:-1].unsqueeze(0)
                batch_h = h[:-1].transpose(0, 1).unsqueeze(0)
                batch_target = x[1:]

                yield (batch_x, batch_h), batch_target

        # re-shuffle
        if shuffle:
            idx = np.random.permutation(n_files)
            wav_list = [wav_list[i] for i in idx]
            feat_list = [feat_list[i] for i in idx]


def save_checkpoint(checkpoint_dir, model, optimizer, iterations):
    """FUNCTION TO SAVE CHECKPOINT"""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iterations": iterations}
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-%d.pkl" % iterations)
    logging.info("%d-iter checkpoint created." % iterations)


class BackgroundGenerator(threading.Thread):
    """BACKGROUND GENERATOR"""
    def __init__(self, generator, max_size=16):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_size)
        self.generator = generator
        self.stop = False
        self.setDaemon(True)
        self.start()

    def run(self):
        for item in self.generator:
            if self.stop:
                break
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", required=True,
                        type=str, help="directory including wav files")
    parser.add_argument("--feat_dir", required=True,
                        type=str, help="directory including aux feat files")
    parser.add_argument("--exp_dir", required=True,
                        type=str, help="directory to save the model")
    parser.add_argument("--stats", required=True,
                        type=str, help="hdf5 file including statistics")
    parser.add_argument("--n_quantize", default=256,
                        type=int, help="number of quantization")
    parser.add_argument("--n_aux", default=28,
                        type=int, help="number of dimension of aux feats")
    parser.add_argument("--n_resch", default=512,
                        type=int, help="number of channels of residual output")
    parser.add_argument("--n_skipch", default=256,
                        type=int, help="number of channels of skip output")
    parser.add_argument("--dilation_depth", default=10,
                        type=int, help="depth of dilation")
    parser.add_argument("--n_repeat", default=3,
                        type=int, help="number of repeating of dilation")
    parser.add_argument("--kernel_size", default=2,
                        type=int, help="kerne size of dilated causal convolution")
    parser.add_argument("--lr", default=1e-3,
                        type=float, help="learning rate")
    parser.add_argument("--batch_size", default=20000,
                        type=int, help="number of iterations")
    parser.add_argument("--n_iters", default=200000,
                        type=int, help="number of iterations")
    parser.add_argument("--checkpoint", default=25000,
                        type=int, help="how frequent saving model")
    parser.add_argument("--interval", default=1000,
                        type=int, help="log interval")
    parser.add_argument("--resume", default=None,
                        type=str, help="model path to restart training")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")
    args = parser.parse_args()

    # set log level
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')

    # define network
    model = WaveNet(n_quantize=args.n_quantize,
                    n_aux=args.n_aux,
                    n_resch=args.n_resch,
                    n_skipch=args.n_skipch,
                    dilation_depth=args.dilation_depth,
                    n_repeat=args.n_repeat,
                    kernel_size=args.kernel_size)
    logging.info(model)
    model.apply(initialize)
    model.cuda()

    # define loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # define generator
    scaler = StandardScaler()
    scaler.mean_ = read_hdf5(args.stats, "/mean")
    scaler.scale_ = read_hdf5(args.stats, "/scale")
    wav_transform = transforms.Compose([
        lambda x: encode_mu_law(x, args.n_quantize),
        lambda x: Variable(torch.from_numpy(x).long().cuda())])
    feat_transform = transforms.Compose([
        lambda x: scaler.transform(x),
        lambda x: Variable(torch.from_numpy(x).float().cuda())])
    generator = custom_generator(
        args.wav_dir, args.feat_dir, model.receptive_field, args.batch_size,
        wav_transform, feat_transform, True, False)
    bg_generator = BackgroundGenerator(generator, 16)

    # train
    iterations = 0
    loss = 0

    # resume
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iterations = checkpoint["iterations"]
        logging.info("restored from %d-iter checkpoint." % iterations)

    start = time.time()
    for i in six.moves.range(iterations, args.n_iters):
        batch = bg_generator.next()
        batch_input, batch_target = batch
        batch_x, batch_h = batch_input
        batch_output = model(batch_x, batch_h)
        batch_loss = criterion(batch_output[model.receptive_field:],
                               batch_target[model.receptive_field:])
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.data[0]

        # report progress
        if (i + 1) % args.interval == 0:
            logging.info("(iter:%d) loss = %.6f (%.3f sec / batch)" % (
                i + 1, loss / args.interval, (time.time() - start) / (i + 1)))
            logging.info("estimated required time = {0.hours:02}:{0.minutes:02}:{0.seconds:02}".format(
                relativedelta(seconds=int((args.n_iters - (i + 1)) * ((time.time() - start) / (i + 1))))))
            loss = 0

        # save model
        if (i + 1) % args.checkpoint == 0:
            save_checkpoint(args.exp_dir, model, optimizer, i + 1)

    # save final model
    save_checkpoint(args.exp_dir, model, optimizer, args.n_iters)
