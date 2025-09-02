import pandas as pd
from model_utils import train_and_save


def load_dataset(path='..\data\dataset.csv'):
    df = pd.read_csv(path, sep='\t')
    texts = df['text'].fillna('').tolist()
    # parse sdg column: expect values like '5' or '1;2;3' or empty
    def parse_sdg(v):
        if pd.isna(v):
            return []
        s = str(v).strip()
        if s == '':
            return []
        # try separators ; or , or whitespace
        for sep in [';', ',', ' ']:
            if sep in s:
                parts = [p.strip() for p in s.split(sep) if p.strip()]
                return parts
        return [s]

    label_lists = [parse_sdg(v) for v in df.get('sdg', df.get('sdg_labels', pd.Series([''] * len(df))))]
    return texts, label_lists


if __name__ == '__main__':
    texts, label_lists = load_dataset()
    clf, vec, mlb = train_and_save(texts, label_lists)
    print('Training complete. Model saved. Classes:', mlb.classes_)
