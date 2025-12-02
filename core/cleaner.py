import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

class Cleaner:
    def __init__(self):
        self.nlp = spacy.load("fr_core_news_sm")
        self.stop_fr = set(stopwords.words("french"))
        self.stop_en = set(stopwords.words("english"))


    @staticmethod
    def clean_basic(text):
        if not text:
            return ""
        return text.lower().strip()


    def clean_text(self,text):
        if not text:
            return ""

        text = self.clean_basic(text)
        doc = self.nlp(text)

        tokens = []
        for tok in doc:
            if any(c.isdigit() for c in tok.text) or "-" in tok.text.upper():
                tokens.append(tok.text.lower())
                continue

            if tok.is_alpha and tok.text not in self.stop_fr and tok.text not in self.stop_en:
                tokens.append(tok.lemma_)

        return " ".join(tokens)


    def clean_category(self, cat):
        if isinstance(cat, list):
            cat = " ".join(cat)
        if not cat:
            return ""
        return self.clean_basic(cat)


    def preprocess_dataframe(self, dfs):
        df = pd.concat(dfs.values(), ignore_index=True)

        df = df[df["id"].notna()]

        df["title_clean"] = df["title"].apply(self.clean_basic)

        df["text_raw_clean"] = df["text"].apply(self.clean_basic)

        df["text_merged"] = df.apply(
            lambda row: row["text_raw_clean"] if row["text_raw_clean"] else row["title_clean"],
            axis=1
        )

        df["text_clean"] = df["text_merged"].apply(self.clean_text)

        df["category_clean"] = df["category"].apply(self.clean_category)

        df = df[df["text_clean"].str.strip() != ""]

        return df