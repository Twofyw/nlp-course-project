from utils.imports import *

class POSTagger:

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
        sentence_splits, poses = [], []
        for sentence in sentences:
            if sentence:
                words = sentence.strip().split('  ')[1:]
                sentence_split, pos = zip(*(o.split('/') for o in words))
                sentence_split = [o.strip() for o in sentence_split] # strip space
                sentence_splits.append(sentence_split)
                poses.append(pos)

        # sentences_space = ' '.join(words_clean)
        # return sentences_space 

        # create segmentation labels
        # prepared_sentences = SegmenterHelper.create_labels(sentences_space)

        # split train val sets
        n_samples = len(sentence_splits)
        n_valid = int(valid_ratio * n_samples)
        
        X = [POSHelper.create_sentence_features(o) for o in sentence_splits[:-n_valid]]
        y = poses[:-n_valid]
        # y = [POSHelper.create_sentence_labels(o) for o in sentence_splits[:-n_valid]]
        X_test = [POSHelper.create_sentence_features(o) for o in sentence_splits[-n_valid:]]
        y_test = poses[-n_valid:]
        # y_test = [POSHelper.create_sentence_labels(o) for o in sentence_splits[-n_valid:]]

        # X = SegmenterHelper.create_sentence_features(prepared_sentences[:-n_valid])
        # y = SegmenterHelper.create_sentence_labels(prepared_sentences[:-n_valid])
        # X_test = SegmenterHelper.create_sentence_features(prepared_sentences[-n_valid:])
        # y_test = SegmenterHelper.create_sentence_labels(prepared_sentences[-n_valid:])

        # save features
        save_name = Path(dir_features)/('pos' + Path(fn_corpora).stem + '_features')
        np.savez_compressed(save_name, X=X, y=y, X_test=X_test, y_test=y_test)
        print(f'Features saved to {save_name}.')

    def train(self, fn_features:str, fn_weights:str):
        """
        dir_features: path to the generated features
        fn_weights: path to save the weights of the CRF model
        """
        features = np.load(fn_features)
        X, y = features['X'], features['y']

        printable = set(string.printable)
        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(X, y):
            yseq = [''.join(filter(lambda x: x in string.printable, o)).strip() for o in yseq]
            trainer.append(xseq, yseq)
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

        y_pred = [tagger.tag(o) for o in X_test]

        y_test_flat = np.array([o for y in y_test for o in y])
        y_score_flat = np.array([o for y in y_pred for o in y])
        classes = np.unique(np.array(y_test_flat))
        y_test_flat, y_score_flat = (label_binarize(o, classes=classes) for o in (y_test_flat, y_score_flat))
        n_classes = Y.shape[1]
        precision, recall, average_precision = POSHelper.eval_mult_class(y_test_flat, y_score_flat)

        POSHelper.plot_mult_class(precision, recall, average_precision)

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
                'char=' + sentence[i] 
                ]

        if i >= 1:
            features.extend([
                'char-1=' + sentence[i-1],
                'char-1:0=' + sentence[i-1] + sentence[i],
                ])
        else:
            features.append("BOS")

        if i >= 2:
            features.extend([
                'char-2=' + sentence[i-2],
                'char-2:0=' + sentence[i-2] + sentence[i-1] + sentence[i],
                'char-2:-1=' + sentence[i-2] + sentence[i-1],
                ])

        if i >= 3:
            features.extend([
                'char-3:0=' + sentence[i-3] + sentence[i-2] + sentence[i-1] + sentence[i],
                'char-3:-1=' + sentence[i-3] + sentence[i-2] + sentence[i-1],
                ])


        if i + 1 < len(sentence):
            features.extend([
                'char+1=' + sentence[i+1],
                'char:+1=' + sentence[i] + sentence[i+1],
                ])
        else:
            features.append("EOS")

        if i + 2 < len(sentence):
            features.extend([
                'char+2=' + sentence[i+2],
                'char:+2=' + sentence[i] + sentence[i+1] + sentence[i+2],
                'char+1:+2=' + sentence[i+1] + sentence[i+2],
                ])

        if i + 3 < len(sentence):
            features.extend([
                'char:+3=' + sentence[i] + sentence[i+1] + sentence[i+2]+ sentence[i+3],
                'char+1:+3=' + sentence[i+1] + sentence[i+2] + sentence[i+3],
                ])

        return features

    @classmethod
    def create_sentence_features(cls, prepared_sentence):
        return [cls.create_char_features(prepared_sentence, i) for i in range(len(prepared_sentence))]

    @classmethod
    def create_sentence_labels(cls, prepared_sentence):
        return [str(part[1]) for part in prepared_sentence]

class POSHelper:
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
                'char=' + sentence[i] 
                ]

        if i >= 1:
            features.extend([
                'char-1=' + sentence[i-1],
                'char-1:0=' + sentence[i-1] + sentence[i],
                ])
        else:
            features.append("BOS")

        if i >= 2:
            features.extend([
                'char-2=' + sentence[i-2],
                'char-2:0=' + sentence[i-2] + sentence[i-1] + sentence[i],
                'char-2:-1=' + sentence[i-2] + sentence[i-1],
                ])

        if i >= 3:
            features.extend([
                'char-3:0=' + sentence[i-3] + sentence[i-2] + sentence[i-1] + sentence[i],
                'char-3:-1=' + sentence[i-3] + sentence[i-2] + sentence[i-1],
                ])


        if i + 1 < len(sentence):
            features.extend([
                'char+1=' + sentence[i+1],
                'char:+1=' + sentence[i] + sentence[i+1],
                ])
        else:
            features.append("EOS")

        if i + 2 < len(sentence):
            features.extend([
                'char+2=' + sentence[i+2],
                'char:+2=' + sentence[i] + sentence[i+1] + sentence[i+2],
                'char+1:+2=' + sentence[i+1] + sentence[i+2],
                ])

        if i + 3 < len(sentence):
            features.extend([
                'char:+3=' + sentence[i] + sentence[i+1] + sentence[i+2]+ sentence[i+3],
                'char+1:+3=' + sentence[i+1] + sentence[i+2] + sentence[i+3],
                ])

        return features

    @classmethod
    def eval_mult_class(cls, Y_test, y_score):
	# For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                    y_score[:, i])
            average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                y_score.ravel())
        average_precision["micro"] = average_precision_score(Y_test, y_score,
                average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'
                .format(average_precision["micro"]))
        return precision, recall, average_precision

    def plot_mult_class(cls, precision, recall, average_precision):
        plt.figure()
        plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
                where='post')
        step_kwargs = ({'step': 'post'}
                if 'step' in signature(plt.fill_between).parameters
                else {})
        plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
                'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
                .format(average_precision["micro"]))

    @classmethod
    def create_sentence_features(cls, prepared_sentence):
        return [cls.create_char_features(prepared_sentence, i) for i in range(len(prepared_sentence))]

    @classmethod
    def create_sentence_labels(cls, prepared_sentence):
        return [str(part[1]) for part in prepared_sentence]



if __name__ == '__main__':
    Fire(Segmenter)
