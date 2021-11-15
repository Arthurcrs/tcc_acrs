from sklearn.base import BaseEstimator, TransformerMixin

class GwasFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extractor for low importance gwas features."""

    def __init__(self, x_locus_tags, gwas_df, gwas_cut_off=0.1):
        """Class constructor."""
        self.gwasDF = gwas_df
        self.gwasCutoff = gwas_cut_off
        self.xLocusTags = x_locus_tags

    def transform(self, x):
        """Transform method."""
        x = self.cut_off(x)

        return x

    def fit(self):
        """Fit method."""
        if self.gwasDF is not None:
            count = 0
            for i in self.gwasDF['p_vals']:
                if i <= self.gwasCutoff:
                    drop = self.gwasDF['LOCUS_TAG'][count]
                    self.gwasDF = self.gwasDF[self.gwasDF['LOCUS_TAG'] != drop]
                count += 1

        return self

    def cut_off(self, x):
        """Cut off low importance features."""
        for i in self.gwasDF['LOCUS_TAG']:
            if i in x.columns:
                x = x.drop(columns=i, axis=1)
        return x
