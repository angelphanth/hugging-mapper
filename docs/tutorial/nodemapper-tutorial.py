# %% [markdown]
# # NodeMapper tutorial
#
# Returning node ids based on similarity of text embeddings.
#
# Start by importing `NodeMapper`

# %%
# An example dataframe
import pandas as pd

# %%
# from hugger import *
from hugger.mapper import NodeMapper

# %% [markdown]
# Demo data for the tutorial


# generate data
ids = ["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"]
texts = [
    "They are happy",
    "I would like to order a doughnut",
    "The grass is green",
    "They are sad",
    "Have you poured the foundation?",
    "I am feeling grey",
    "blue",
    "home",
]
# to dataframe
df = pd.DataFrame({"id": ids, "text": texts})

# %% [markdown]
# Initializing `NodeMapper` will
# - load the given huggingface model
# - generate embeddings for the text column
# - creating a dictionary of the node ids : text embeddings

# %%
# init
mapper = NodeMapper(
    df=df,
    text_col="text",
    id_col="id",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# %% [markdown]
# Like `HuggingMapper` can simply get embeddings for given text

# %%
# generate embedding for a single text
embedding = mapper.embed_text("Good morning")
print(embedding.shape)

# generate embeddings for a list of texts
embeddings = mapper.embed_text(["Hello world", "Good evening", "Lunch time!"])
print(embeddings.shape)

# %% [markdown]
# But the main purpose of `NodeMapper` is to find similar texts and their corresponding ids

# %%
# retrieve those most similar to given text, above threshold
mapper.get_similar("concrete", threshold=0)  # threshold 0 returns all

# %%
# retrieve top match, above threshold
print(mapper.get_match("joyful", threshold=0.4), "\n")
print(mapper.get_match("we are crying", threshold=0.4), "\n")
print(mapper.get_match("eatting a donut", threshold=0.4), "\n")

# %%
# retrieve top k matches, above threshold
print(mapper.get_similar("yellow", threshold=0.3, top_k=2), "\n")
print(mapper.get_similar("laughter", top_k=3), "\n")
