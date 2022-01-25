""" Manage beam search info structure.
    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import torch
import numpy as np
from src.utils import config
from src.utils.decode.beam import Beam


class Translator(object):
    """ Load with trained model and handle the beam search """

    def __init__(self, model, lang):

        self.model = model
        self.lang = lang
        self.vocab_size = lang.n_words
        self.beam_size = config.beam_size
        self.device = config.device

    def beam_search(self, src_seq, max_dec_step, emotion_classifier="built_in"):
        """ Translation work in one batch """

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            """ Indicate the position of an instance in a tensor. """
            return {
                inst_idx: tensor_position
                for tensor_position, inst_idx in enumerate(inst_idx_list)
            }

        def collect_active_part(
            beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm
        ):
            """ Collect tensor parts associated to active instances. """

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
            src_seq, encoder_db, src_enc, inst_idx_to_position_map, active_inst_idx_list
        ):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [
                inst_idx_to_position_map[k] for k in active_inst_idx_list
            ]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(
                src_seq, active_inst_idx, n_prev_active_inst, n_bm
            )
            active_src_enc = collect_active_part(
                src_enc, active_inst_idx, n_prev_active_inst, n_bm
            )

            active_encoder_db = None

            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
                active_inst_idx_list
            )

            return (
                active_src_seq,
                active_encoder_db,
                active_src_enc,
                active_inst_idx_to_position_map,
            )

        def beam_decode_step(
            inst_dec_beams,
            len_dec_seq,
            src_seq,
            enc_output,
            v,
            inst_idx_to_position_map,
            n_bm,
            enc_batch_extend_vocab,
            extra_zeros,
            mask_src,
            encoder_db,
            mask_transformer_db,
            DB_ext_vocab_batch,
        ):
            """ Decode and update beam status, and then return active beam idx """

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [
                    b.get_current_state() for b in inst_dec_beams if not b.done
                ]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_atten(inst_dec_beams):
                atten = [
                    self.attention_parameters[i]
                    for i, b in enumerate(inst_dec_beams)
                    if not b.done
                ]
                atten = torch.stack(atten).to(self.device)
                atten = atten.view(-1, self.len_program, 1, 1)
                atten = atten.repeat(1, n_bm, 1, 1).view(-1, self.len_program, 1, 1)
                return atten

            def prepare_beam_target(inst_dec_beams):
                tgt_program = [
                    torch.Tensor(batch["target_program"])[i]
                    for i, b in enumerate(inst_dec_beams)
                    if not b.done
                ]
                tgt_program = torch.stack(tgt_program).to(self.device)
                tgt_program = tgt_program.view(-1, self.len_program)
                tgt_program = tgt_program.repeat(1, n_bm).view(-1, self.len_program)
                return tgt_program

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(
                    1, len_dec_seq + 1, dtype=torch.long, device=self.device
                )
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(
                    n_active_inst * n_bm, 1
                )
                return dec_partial_pos

            def predict_word(
                dec_seq,
                dec_pos,
                src_seq,
                enc_output,
                n_active_inst,
                n_bm,
                enc_batch_extend_vocab,
                extra_zeros,
                mask_src,
                encoder_db,
                mask_transformer_db,
                atten,
            ):
                ## masking
                mask_trg = dec_seq.data.eq(config.PAD_idx).unsqueeze(1)

                # mask_src_chosen = mask_src if config.emo_input == "cross_att" else mask_src_with_emo

                mask_src = torch.cat([mask_src[0].unsqueeze(0)] * mask_trg.size(0), 0)
                # mask_src_chosen = torch.cat([mask_src_chosen[0].unsqueeze(0)]*mask_trg.size(0),0)

                #                 if config.decoder == 'mul':
                #                     dec_output, attn_dist = self.model.decoder(self.model.embedding(dec_seq), enc_output, (mask_src,mask_trg), atten)
                #                 elif config.decoder == 'single':
                #                     dec_output, attn_dist = self.model.decoder(self.model.embedding(dec_seq), enc_output, (mask_src,mask_trg))
                # Code copied from main_mimic_vader.py
                if config.decoder == "mul":
                    dec_output, attn_dist = self.model.decoder(
                        self.model.embedding(dec_seq),
                        encoder_outputs,
                        (mask_src, mask_trg),
                        atten,
                    )
                elif config.decoder == "single":
                    dec_output, attn_dist = self.model.decoder(
                        self.model.embedding(dec_seq), v, v, (mask_src, mask_trg)
                    )

                db_dist = None

                prob = self.model.generator(
                    dec_output,
                    attn_dist,
                    enc_batch_extend_vocab,
                    extra_zeros,
                    1,
                    True,
                    attn_dist_db=db_dist,
                )
                # prob = F.log_softmax(prob,dim=-1) #fix the name later
                word_prob = prob[:, -1]
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
                return word_prob

            def collect_active_inst_idx_list(
                inst_beams, word_prob, inst_idx_to_position_map
            ):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(
                        word_prob[inst_position]
                    )
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]
                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)
            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(
                dec_seq,
                dec_pos,
                src_seq,
                enc_output,
                n_active_inst,
                n_bm,
                enc_batch_extend_vocab,
                extra_zeros,
                mask_src,
                encoder_db,
                mask_transformer_db,
                atten=self.attention_parameters,
            )

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map
            )

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [
                    inst_dec_beams[inst_idx].get_hypothesis(i)
                    for i in tail_idxs[:n_best]
                ]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            # -- Encode
            batch = src_seq
            (
                enc_batch,
                _,
                _,
                enc_batch_extend_vocab,
                extra_zeros,
                _,
                _,
            ) = get_input_from_batch(src_seq)
            emotions = batch["program_label"]

            context_emo = [
                self.model.positive_emotions[0]
                if d["compound"] > 0
                else self.model.negative_emotions[0]
                for d in batch["context_emotion_scores"]
            ]
            context_emo = torch.Tensor(context_emo).to(config.device)

            if config.noam:
                self.model.optimizer.optimizer.zero_grad()
            else:
                self.model.optimizer.zero_grad()
            ## Encode
            mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)

            emb_mask = self.model.embedding(batch["mask_input"])
            encoder_outputs = self.model.encoder(
                self.model.embedding(enc_batch) + emb_mask, mask_src
            )

            q_h = (
                torch.mean(encoder_outputs, dim=1)
                if config.mean_query
                else encoder_outputs[:, 0]
            )

            x = self.model.s_weight(q_h)
            logit_prob = torch.softmax(
                torch.matmul(x, self.model.emoji_embedding.weight.transpose(0, 1)),
                dim=-1,
            )
            emo_pred = torch.argmax(logit_prob, dim=-1)

            if emotion_classifier == "vader":
                context_emo = [
                    self.model.positive_emotions[0]
                    if d["compound"] > 0
                    else self.model.negative_emotions[0]
                    for d in batch["context_emotion_scores"]
                ]
                context_emo = torch.Tensor(context_emo).to(config.device)
                (
                    emotions_mimic,
                    emotions_non_mimic,
                    mu_p,
                    logvar_p,
                    mu_n,
                    logvar_n,
                ) = self.model.vae_sampler(q_h, context_emo, self.model.emoji_embedding)
            elif emotion_classifier == None:
                (
                    emotions_mimic,
                    emotions_non_mimic,
                    mu_p,
                    logvar_p,
                    mu_n,
                    logvar_n,
                ) = self.vae_sampler(
                    q_h, batch["program_label"], self.model.emoji_embedding
                )
            elif emotion_classifier == "built_in":
                (
                    emotions_mimic,
                    emotions_non_mimic,
                    mu_p,
                    logvar_p,
                    mu_n,
                    logvar_n,
                ) = self.model.vae_sampler(q_h, emo_pred, self.model.emoji_embedding)

            m_out = self.model.emotion_input_encoder_1(
                emotions_mimic.unsqueeze(1).to(config.device), encoder_outputs, mask_src
            )
            m_tilde_out = self.model.emotion_input_encoder_2(
                emotions_non_mimic.unsqueeze(1).to(config.device),
                encoder_outputs,
                mask_src,
            )
            # m_out = self.model.emotion_input_encoder_1(emotions[:, 0].unsqueeze(1), encoder_outputs,mask_src)
            # m_tilde_out = self.model.emotion_input_encoder_2(emotions[:, 1].unsqueeze(1), encoder_outputs,mask_src)

            if config.emo_combine == "att":
                v = self.model.cdecoder(encoder_outputs, m_out, m_tilde_out, mask_src)
            #                     v = self.model.cdecoder(encoder_outputs, m_out, m_tilde_out, mask_src_chosen)
            elif config.emo_combine == "gate":
                #                     v, z, z_tilde = self.model.cdecoder(m_out, m_tilde_out)
                v = self.model.cdecoder(m_out, m_tilde_out)
                # z_target = target_emo_scores.unsqueeze(-1).unsqueeze(-1)
                # z_tilde_target = 1 - z_target
            src_enc = encoder_outputs

            encoder_db = None
            mask_transformer_db = None
            DB_ext_vocab_batch = None

            ## Attention over decoder
            #             q_h = torch.mean(src_enc,dim=1) if config.mean_query else src_enc[:,0]
            # q_h = src_enc[:,0]
            #             logit_prob = self.model.decoder_key(q_h)
            x = self.model.s_weight(q_h)
            logit_prob = torch.softmax(
                torch.matmul(x, self.model.emoji_embedding.weight.transpose(0, 1)),
                dim=-1,
            )

            if config.topk > 0:
                k_max_value, k_max_index = torch.topk(logit_prob, config.topk)
                a = np.empty([logit_prob.shape[0], self.model.decoder_number])
                a.fill(float("-inf"))
                mask = torch.Tensor(a).to(config.device)
                k_max_index.to(config.device)
                logit_prob = mask.scatter_(1, k_max_index.long(), k_max_value)

            attention_parameters = self.model.attention_activation(logit_prob)

            if config.oracle:
                attention_parameters = self.model.attention_activation(
                    torch.FloatTensor(src_seq["target_program"]) * 1000
                ).to(config.device)
            self.attention_parameters = attention_parameters.unsqueeze(-1).unsqueeze(-1)

            # -- Repeat data for beam search
            n_bm = self.beam_size
            n_inst, len_s, d_h = src_enc.size()
            _, self.len_program, _, _ = self.attention_parameters.size()
            src_seq = enc_batch.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
                active_inst_idx_list
            )

            # -- Decode
            for len_dec_seq in range(1, max_dec_step + 1):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams,
                    len_dec_seq,
                    src_seq,
                    src_enc,
                    v,
                    inst_idx_to_position_map,
                    n_bm,
                    enc_batch_extend_vocab,
                    extra_zeros,
                    mask_src,
                    encoder_db,
                    mask_transformer_db,
                    DB_ext_vocab_batch,
                )

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                (
                    src_seq,
                    encoder_db,
                    src_enc,
                    inst_idx_to_position_map,
                ) = collate_active_info(
                    src_seq,
                    encoder_db,
                    src_enc,
                    inst_idx_to_position_map,
                    active_inst_idx_list,
                )

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)

        ret_sentences = []
        for d in batch_hyp:
            ret_sentences.append(
                " ".join([self.model.vocab.index2word[idx] for idx in d[0]]).replace(
                    "EOS", ""
                )
            )

        return ret_sentences  # , batch_scores


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = seq_range_expand
    seq_range_expand.to(config.device)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def get_input_from_batch(batch):
    enc_batch = batch["input_batch"]
    enc_lens = batch["input_lengths"]
    batch_size, max_enc_len = enc_batch.size()
    assert enc_lens.size(0) == batch_size

    enc_padding_mask = sequence_mask(enc_lens, max_len=max_enc_len).float()

    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = batch["input_ext_vocab_batch"]
        # max_art_oovs is the max over all the article oov list in the batch
        if batch["max_art_oovs"] > 0:
            extra_zeros = torch.zeros((batch_size, batch["max_art_oovs"]))

    c_t_1 = torch.zeros((batch_size, 2 * config.hidden_dim)).to(config.device)

    coverage = None
    if config.is_coverage:
        coverage = torch.zeros(enc_batch.size()).to(config.device)

    if enc_batch_extend_vocab is not None:
        enc_batch_extend_vocab.to(config.device)

    if extra_zeros is not None:
        extra_zeros.to(config.device)

    return (
        enc_batch,
        enc_padding_mask,
        enc_lens,
        enc_batch_extend_vocab,
        extra_zeros,
        c_t_1,
        coverage,
    )
