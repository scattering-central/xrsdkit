from collections import OrderedDict

import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans

from .xrsd_model import XRSDModel
from ..tools import profiler


class Classifier(XRSDModel):
    """Class for models that classify attributes of material systems."""

    def __init__(self,label,yml_file):
        super(Classifier,self).__init__(label, yml_file)
        self.hyperparam_grid = dict(
            C = np.logspace(-1,3,num=15,endpoint=True,base=10.)
            )
        self.sgd_hyperparam_grid = dict(
            #alpha = [1.E-5,1.E-4,1.E-3,1.E-2,1.E-1],
            #l1_ratio = [0., 0.15, 0.5, 0.85, 1.0]
            alpha = np.logspace(-1,2,num=9,endpoint=True,base=10.),
            l1_ratio = np.linspace(0.,1.,num=7,endpoint=True) 
            )

    def build_model(self,model_hyperparams={}):
        penalty='l2'
        if 'penalty' in model_hyperparams: penalty = model_hyperparams['penalty']
        C = 1.
        if 'C' in model_hyperparams: C = model_hyperparams['C']
        solver = 'lbfgs'
        if 'solver' in model_hyperparams: solver = model_hyperparams['solver']
        new_model = linear_model.LogisticRegression(penalty=penalty, C=C, 
            class_weight='balanced', solver=solver, max_iter=100000)
        return new_model

    def build_sgd_model(self,model_hyperparams={}):
        alpha = 1.E-4
        if 'alpha' in model_hyperparams: alpha = model_hyperparams['alpha']
        l1_ratio = 0.15 
        if 'l1_ratio' in model_hyperparams: l1_ratio = model_hyperparams['l1_ratio']
        new_model = linear_model.SGDClassifier(alpha=alpha, loss='log', penalty='elasticnet', l1_ratio=l1_ratio,
                max_iter=1000000, class_weight='balanced', tol=1e-10, eta0 = 0.001, learning_rate='adaptive')
        return new_model

    def classify(self, sample_features):
        """Classify the model target for a sample.

        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of xrsdkit.tools.profiler.profile_pattern().

        Returns
        -------
        cls : object 
            Predicted classification value for self.target, given `sample_features`
        cert : float or None
            the certainty of the prediction
            (For models that are not trained, cert=None)
        """
        if self.trained:
            feature_array = np.array(list(sample_features.values())).reshape(1,-1)
            feature_idx = [k in self.features for k in sample_features.keys()]
            x = self.scaler.transform(feature_array)[:, feature_idx]
            cls = self.model.predict(x)[0]
            cert = max(self.model.predict_proba(x)[0])
            return cls, cert
        else:
            return (self.default_val,0.)

    def cv_report(self,data,y_true,y_pred):
        all_labels = data[self.target].unique().tolist()
        y_true_all = []
        y_pred_all = []
        for gid,yt in y_true.items():
            y_true_all.extend(yt)
        for gid,yp in y_pred.items():
            y_pred_all.extend(yp)
        cm = confusion_matrix(y_true_all, y_pred_all, all_labels)
        result = dict(
            all_labels = all_labels,
            confusion_matrix = str(cm),
            f1_macro = f1_score(y_true_all,y_pred_all,labels=all_labels,average='macro'),
            precision = precision_score(y_true_all, y_pred_all, average='macro'),
            recall = recall_score(y_true_all, y_pred_all, average='macro'),
            accuracy = accuracy_score(y_true_all, y_pred_all, sample_weight=None)
            )
        #print('f1: {}'.format(result['f1_score']))
        result['minimization_score'] = -1*result['f1_macro']
        #result['minimization_score'] = -1*result['accuracy']
        return result

    def group_by_pc1(self,dataframe,feature_names,n_groups=5):
        label_cts = dataframe[self.target].value_counts()
        if len(label_cts) < 2: return False
        labels = list(label_cts.keys())
        for l in labels:
            if label_cts[l] < n_groups:
                # this label cannot be spread across the groups:
                # remove it from the model entirely 
                label_cts.pop(l)
        groups_possible = self._diverse_groups_possible(dataframe,n_groups,len(label_cts.keys()))
        if not groups_possible: return False

        group_ids = range(1,n_groups+1)
        for label in label_cts.keys():
            lidx = dataframe.loc[:,self.target]==label
            ldata = dataframe.loc[lidx,feature_names]
            pc1 = PCA(n_components=1)
            ldata_pc = pc1.fit_transform(ldata).ravel()
            pc_rank = np.argsort(ldata_pc)
            lgroups = np.zeros(ldata.shape[0])
            gp_size = int(round(ldata.shape[0]/n_groups))
            for igid,gid in enumerate(group_ids):
                lgroups[pc_rank[igid*gp_size:(igid+1)*gp_size]] = int(gid)
            dataframe.loc[lidx,'group_id'] = lgroups
        return True

    def print_confusion_matrix(self):
        result = ''
        matrix = self.cross_valid_results['confusion_matrix'].split('\n')
        for ilabel,label in enumerate(self.cross_valid_results['all_labels']):
            result += (matrix[ilabel]+"  "+str(label)+'\n')
        return result

    def print_CV_report(self):
        """Return a string describing the model's cross-validation metrics.

        Returns
        -------
        CV_report : str
            string with formatted results of cross validation.
        """
        # TODO: document the computation of these metrics, 
        # then refer to documentation in this report
        # TODO: add sample_ids and groupings to this report 
        # TODO: add feature names to this report
        CV_report = 'Cross validation results for {} Classifier\n\n'.format(self.target) + \
            'Confusion matrix:\n' + \
            self.print_confusion_matrix()+'\n\n' + \
            'F1 score: {}\n\n'.format(
            self.cross_valid_results['f1_macro']) + \
            'Precision: {}\n\n'.format(
            self.cross_valid_results['precision']) + \
            'Recall: {}\n\n'.format(
            self.cross_valid_results['recall']) + \
            'Accuracy: {}\n\n'.format(
            self.cross_valid_results['accuracy']) 
        return CV_report

