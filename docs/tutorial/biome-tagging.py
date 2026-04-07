# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: hugging-mapper
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Biome ontology tagging demo
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angelphanth/hugging-mapper/blob/main/docs/tutorial/biome-tagging.ipynb)
#

# %%
# uncomment if colab
# #!pip install pandas hugging-mapper

# %% [markdown]
#
# <details>
# <summary style=color:orange> 
# <u>Click here</u> for more info on: <br><br><i><b>"Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads."</i></b>
# </summary>
# <h1></h1>
#
# **What is an HF_TOKEN / huggingface user access token?**
# > *[User Access Tokens are the preferred way to authenticate an application or notebook to Hugging Face services. You can manage your access tokens in your settings.](huggingface.co/docs/hub/en/security-tokens#what-are-user-access-tokens)*
#
# If you have an `HF_TOKEN` you can add it to your environment variables, repository secrets, and/or you can access it in your venv by saving the HF_TOKEN in an *.env* file and then loading it via package [`python-dotenv`](https://pypi.org/project/python-dotenv/).
#
# For example: 
# 1. More information for getting a Huggingface user token: [their docs](https://huggingface.co/docs/hub/en/security-tokens)
# 2. Save to "HF_TOKEN" variable
#
#     Example *.env* file: 
#     ```bash
#     HF_TOKEN=hf***...
#     ```
# 3. Access the .env variables via python-dotenv 
#
#     e.g. 
#     ```python
#     from dotenv import load_dotenv
#     load_dotenv()
#     ```
#
# <h1></h1>
# </details>
# <br>

# %%
# from dotenv import load_dotenv
# load_dotenv()

# %% [markdown]
# ## Prep the data
# First we can encode each of the texts/terms in the ontology that we will wish to search against e.g. 

# %% [markdown]
# ### The [GOLD Biome Ontology](https://bioportal.bioontology.org/ontologies/GOLDTERMS) 
#
# We will read in as pandas dataframe 

# %%
import pandas as pd

gold = pd.read_csv("https://github.com/cmungall/gold-ontology/raw/refs/heads/main/gold_definitions.csv")

gold.head()

# %% [markdown]
# we will generate embeddings for the text in the 'label' column using NodeMapper. calling nodemapper will automatically start embedding the text. 
#
# You can try out different `model_name`s from [hugging face](https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=trending)

# %%
from hugger.mapper import NodeMapper

# from huggingface
model_name = "sentence-transformers/all-MiniLM-L6-v2"

mapper = NodeMapper(
    df=gold.iloc[1:],
    text_col="label",
    id_col="id",
    model_name=model_name,
)

# you can access embeddings via the mapping_embeddings attribute,
# which is a dictionary mapping from node ID to embedding tensor
# mapper.mapping_embeddings

# or as a df
mapper.embeddings_df.head()

# %% [markdown]
# the embeddings are stored in the mapping_embeddings attribute, which is a dictionary mapping from node ID to embedding tensor. 
#
# Of course a better way to visualize in 2D, for which can do a quick plot with the `plot_tsne()`

# %%
mapper.plot_tsne(title="t-SNE of GOLD embeddings")

# %% [markdown]
# ### The searching texts
#
# - In this example we will use the text in sample metadata such as names, descriptions and project title. 
#
# - for each sample we will find the most semanticly similar gold biome term(s)
#
# - by comparing the encoded gold biome term vectors vs. the encoded sample text metadata vector (i.e., concatenated project name, sample title, sample description.. etc)
#
# - most similar will be based on best cosine similarity between vectors

# %%
# read in sample metadata
# TODO replace with repo link
df = pd.read_csv("assets/biosamples-marine-sample.tsv", sep="\t")
# quick replace of semicolons
df = df.rename(columns={df.columns[1]: "text"})
df['text'] = df['text'].str.replace(";", " ")
df['text'] = df['text'].str.replace("_", " ")
df['text'] = df['text'].str.replace("-", " ")
# sanity check
df.head()

# %% [markdown]
# ## Search
#
# For this demo we will take an even smaller subset of the df as original was 700K long

# %%
# get subset
subset = df.sample(50, random_state=42, ignore_index=True)
print("New subset shape:", subset.shape)

# %%
# init
trial = {}
counter = 0
for i in range(len(subset)):
    # init sample dict
    trial[i] = {}
    # sample accession and text
    trial[i]['sample_accession'] = subset.loc[i, "sample_accession"]
    trial[i]['text'] = subset.loc[i, "text"]
    # get top 3 predictions
    top_ks = mapper.get_similar(trial[i]['text'], top_k=3)
    top_k_ids = list(top_ks.keys())
    for j, k in enumerate(top_k_ids, start=1):
        trial[i][f'predicted_{j}'] = top_ks[k]['text']
        trial[i][f'score_{j}'] = top_ks[k]['score']
    # also get actual tag for comparison
    trial[i]['actual'] = subset.loc[i, "tag"]
    # counter for progress tracking
    counter += 1
    if counter % 10 == 0:
        # verbose
        print(f"Processed {counter} examples")
        # write to json
        # with open("assets/trial_results.json", "w") as f:
        #     json.dump(trial, f, indent=4)

# %%
# check it out
result_df = pd.DataFrame.from_dict(trial, orient="index")
result_df.head()

# save to tsv
#result_df.to_csv(f"assets/gold-trial-{model_name.split('/')[-1]}.tsv", sep="\t")
