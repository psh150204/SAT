def sen2seq(voca, sentence):
    # input - voca : dictionary / sentence : a list of words
    return [voca['<sos>']] + [voca[x.lower()] for x in sentence] + [voca['<eos>']]

def seq2sen(batch, mapping):
    sen_list = []

    for seq in batch:
        if 1 in seq:
            end = seq.index(1)
        else:
            end = -1
        
        seq_strip = seq[:end]
        sen = [list(mapping.keys())[list(mapping.values()).index(token)] for token in seq_strip[1:]]
        sen_list.append(sen)

    return sen_list

from nltk.translate.bleu_score import corpus_bleu

hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which','ensures', 'that', 'the', 'military', 'always','obeys', 'the', 'commands', 'of', 'the', 'party']
ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that','ensures', 'that', 'the', 'military', 'will', 'forever','heed', 'Party', 'commands']
ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which','guarantees', 'the', 'military', 'forces', 'always','being', 'under', 'the', 'command', 'of', 'the', 'Party']
ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the','army', 'always', 'to', 'heed', 'the', 'directions','of', 'the', 'party']
hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was','interested', 'in', 'world', 'history']
ref2a = ['he', 'was', 'interested', 'in', 'world', 'history','because', 'he', 'read', 'the', 'book']
list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
hypotheses = [hyp1, hyp2]

bleu = corpus_bleu(list_of_references, hypotheses)
bleu_1gram = corpus_bleu(list_of_references, hypotheses, weights = (1,0,0,0))
bleu_2gram = corpus_bleu(list_of_references, hypotheses, weights = (0,1,0,0))
bleu_3gram = corpus_bleu(list_of_references, hypotheses, weights = (0,0,1,0))
bleu_4gram = corpus_bleu(list_of_references, hypotheses, weights = (0,0,0,1))

print(f'BLEU: {bleu:.2f}')
print(f'1-Gram BLEU: {bleu_1gram:.2f}')
print(f'2-Gram BLEU: {bleu_2gram:.2f}')
print(f'3-Gram BLEU: {bleu_3gram:.2f}')
print(f'4-Gram BLEU: {bleu_4gram:.2f}')