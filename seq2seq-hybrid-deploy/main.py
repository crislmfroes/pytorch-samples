import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np

device = torch.device('cpu')

MAX_LENGTH = 10

PAD_token = 0
SOS_token = 1
EOS_token = 2

class Voc(object):
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def addSentence(self, sentence: str):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for key, value in self.word2count.items():
            if value >= min_count:
                keep_words.append(key)
        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index),
            len(keep_words) / len(self.word2index)
        ))
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD",
                           SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        for word in keep_words:
            self.addWord(word)


def normalizeString(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ') if word in voc.word2index] + [EOS_token]

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
            dropout=(0 if n_layers ==1 else dropout),
            bidirectional=True
        )
    
    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,:self.hidden_size:]
        return outputs, hidden


class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size*2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))
    
    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden*energy, dim=2)
    
    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v*energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoder(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoder, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(
            0 if n_layers == 1 else dropout
        ))
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden


class GreedySearchDecoder(torch.jit.ScriptModule):
    def __init__(self, encoder, decoder, decoder_n_layers):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._device = device
        self._SOS_token = SOS_token
        self._decoder_n_layers = decoder_n_layers

    __constants__ = ['_device', '_SOS_token', '_decoder_n_layers']

    @torch.jit.script_method
    def forward(self, input_seq: torch.Tensor, input_length: torch.Tensor, max_length: int):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self._decoder_n_layers]
        decoder_input = torch.ones(1, 1, device=self._device, dtype=torch.long) * self._SOS_token
        all_tokens = torch.zeros([0], device=self._device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self._device)
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(voc, sentence)]
    lengths = torch.Tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        input_sentence = input('> ')
        if input_sentence == 'q' or input_sentence == 'quit':
            break
        input_sentence = normalizeString(input_sentence)
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        print('Bot:', ' '.join(output_words))


def evaluateExample(sentence, encoder, decoder, searcher, voc):
    print ("> " + sentence)
    input_sentence = normalizeString(sentence)
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]


save_dir = os.path.join('data', 'save')
corpus_name = 'cornell movie-dialogs corpus'

model_name = 'cb_model'
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

loadFilename = 'data/4000_checkpoint.tar'

checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']
voc = Voc(corpus_name)
voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')

embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)

encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoder(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
print('Ok!')
encoder.load_state_dict(encoder_sd)
print('Ok!')
decoder.load_state_dict(decoder_sd)
print('Ok!')
encoder = encoder.to(device)
print('Ok!')
decoder = decoder.to(device)
print('Ok!')

encoder.eval()
print('Ok!')
decoder.eval()
print('Ok!')

print('Models built and ready to go!')

test_seq = torch.LongTensor(MAX_LENGTH, 1).random_(0, voc.num_words)
test_seq_length = torch.LongTensor([test_seq.size()[0]])
traced_encoder = torch.jit.trace(encoder, (test_seq, test_seq_length))

test_encoder_outputs, test_encoder_hidden = traced_encoder(test_seq, test_seq_length)
test_decoder_hidden = test_encoder_hidden[:decoder.n_layers]
test_decoder_input = torch.LongTensor(1, 1).random_(0, voc.num_words)


traced_decoder = torch.jit.trace(decoder, (test_decoder_input, test_decoder_hidden, test_encoder_outputs))

scripted_searcher = GreedySearchDecoder(traced_encoder, traced_decoder, decoder.n_layers)

print('scripted_searcher graph:\n', scripted_searcher.graph)

scripted_searcher.save("data/scripted_chatbot.pth")

evaluateInput(traced_encoder, traced_decoder, scripted_searcher, voc)
