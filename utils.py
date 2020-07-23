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