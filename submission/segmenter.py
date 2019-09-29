from utils.imports import *

class Segmenter:

    def create_dataset(self, fn_corpora:str, dir_features:str, valid_ratio=0.2):
        """
        fn_corpora: path to 北大人民日报语料库.txt
        dir_features: path to the directory to save the generated features
        """
        # read raw data
        with open(fn_corpora, 'r', encoding='gbk') as f:
            raw_corpora = f.read()

        # extrat space seperated words from copora
        sentences = raw_corpora.split('\n')
        words = [o for sentence in sentences for o in sentence.split('  ')[1:-1]]
        words_clean = [o.split('/')[0] for o in words]
        sentences_space = ' '.join(words_clean)

        # create labels
        prepared_sentences = SegmenterHelper.create_labels(sentences_space)

        # split train val sets
        n_samples = len(prepared_sentences)
        n_valid = int(valid_ratio * n_samples)
        
        X = SegmenterHelper.create_sentence_features(prepared_sentences[:-n_valid])
        y = SegmenterHelper.create_sentence_labels(prepared_sentences[:-n_valid])
        X_test = SegmenterHelper.create_sentence_features(prepared_sentences[-n_valid:])
        y_test = SegmenterHelper.create_sentence_labels(prepared_sentences[-n_valid:])

        # save features
        Path(dir_features).mkdir(exist_ok=True, parents=True)
        save_name = Path(dir_features)/(Path(fn_corpora).stem + '_features')
        np.savez_compressed(save_name, X=X, y=y, X_test=X_test, y_test=y_test)
        print(f'Features saved to {save_name}.')

    def train(self, fn_features:str, fn_weights:str):
        """
        dir_features: path to the generated features
        fn_weights: path to save the weights of the CRF model
        """
        features = np.load(fn_features)
        X, y = features['X'], features['y']

        trainer = pycrfsuite.Trainer(verbose=False)
        trainer.append(X, y)
        trainer.set_params({
            'c1': 1.0, 
            'c2': 1e-3,
            'max_iterations': 60,
            'feature.possible_transitions': True
            })

        save_name = fn_weights + '.crfsuite'
        trainer.train(save_name)
        print(f'Weights saved to {save_name}.')

    def evaluate(self, fn_features:str, fn_weights:str):
        """
        dir_features: path to the generated features
        fn_weights: path to the weights of the CRF model
        """
        features = np.load(fn_features)
        X_test, y_test = features['X_test'], features['y_test']

        tagger = pycrfsuite.Tagger()
        tagger.open(fn_weights + '.crfsuite')
        
        y_pred = tagger.tag(X_test)
        y_true, y_pred = np.array(y_test, dtype=np.float), np.array(y_pred, dtype=np.float)
        print('average_precision_score', average_precision_score(y_true, y_pred), file=sys.stderr)
        print('precision_recall_fscore_support', precision_recall_fscore_support(y_true, y_pred), file=sys.stderr)

        sentence = []
        it = iter(X_test)
        for y in y_pred:
            if y == 1: sentence.append(' ')
            sentence.append(next(it)[1][-1])

        return ''.join(sentence)


class SegmenterHelper:
    @classmethod
    def create_labels(cls, sentences_space):
        """ Create a list of labels from a space seperated sentence """
        sentences_space_np = np.array(list(sentences_space))
        space_bool = (sentences_space_np == ' ')
        space_idx = np.argwhere(space_bool).squeeze()

        shift = np.arange((space_bool == True).sum())
        b_idx = space_idx - shift
        concatenated = sentences_space.replace(" ", "")
        b_bool = np.zeros(len(concatenated))
        b_bool[b_idx] = 1

        prepared_sentences = list(zip(concatenated, b_bool))
        return prepared_sentences

    @classmethod
    def create_char_features(cls, sentence, i):
        features = [
                'bias',
                'char=' + sentence[i][0] 
                ]

        if i >= 1:
            features.extend([
                'char-1=' + sentence[i-1][0],
                'char-1:0=' + sentence[i-1][0] + sentence[i][0],
                ])
        else:
            features.append("BOS")

        if i >= 2:
            features.extend([
                'char-2=' + sentence[i-2][0],
                'char-2:0=' + sentence[i-2][0] + sentence[i-1][0] + sentence[i][0],
                'char-2:-1=' + sentence[i-2][0] + sentence[i-1][0],
                ])

        if i >= 3:
            features.extend([
                'char-3:0=' + sentence[i-3][0] + sentence[i-2][0] + sentence[i-1][0] + sentence[i][0],
                'char-3:-1=' + sentence[i-3][0] + sentence[i-2][0] + sentence[i-1][0],
                ])


        if i + 1 < len(sentence):
            features.extend([
                'char+1=' + sentence[i+1][0],
                'char:+1=' + sentence[i][0] + sentence[i+1][0],
                ])
        else:
            features.append("EOS")

        if i + 2 < len(sentence):
            features.extend([
                'char+2=' + sentence[i+2][0],
                'char:+2=' + sentence[i][0] + sentence[i+1][0] + sentence[i+2][0],
                'char+1:+2=' + sentence[i+1][0] + sentence[i+2][0],
                ])

        if i + 3 < len(sentence):
            features.extend([
                'char:+3=' + sentence[i][0] + sentence[i+1][0] + sentence[i+2][0]+ sentence[i+3][0],
                'char+1:+3=' + sentence[i+1][0] + sentence[i+2][0] + sentence[i+3][0],
                ])

        return features

    @classmethod
    def create_sentence_features(cls, prepared_sentence):
        return [cls.create_char_features(prepared_sentence, i) for i in range(len(prepared_sentence))]

    @classmethod
    def create_sentence_labels(cls, prepared_sentence):
        return [str(part[1]) for part in prepared_sentence]



if __name__ == '__main__':
    Fire(Segmenter)
