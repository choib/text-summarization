import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pydash
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import (
                                     RecursiveCharacterTextSplitter,
                                     SentenceTransformersTokenTextSplitter,
                                     CharacterTextSplitter,
                                     )
from networkx.algorithms import community
from scipy.spatial.distance import cosine
from config import config
from utilities.custom_logger import CustomLogger

logger = CustomLogger()

@dataclass
class ProcessingPipeline:
    embeddings: Embeddings
    num_of_tokens: Optional[int] = None

    def process_document(self, document: str) -> List[str]:
        """process a long document into list of shorter chunks, where each chunk has a unique cluster

        Args:
            document (str): long text

        Returns:
            List[str]: list of chunks
        """
        paragraphs = self.split_document(document)
        embedding_dict = self.get_embeddings(paragraphs)
        chunks = self.cluster_similar_chunks(embedding_dict, len(paragraphs))
        return chunks

    @staticmethod
    def get_num_of_tokens(text: str) -> int:
        """get No. of tokens in the text"""
        return len(config.TOKENIZER.tokenize(text))

    def is_paragraph(self, txt):
        """filter the paragraph with index"""
        if (re.match(r"^[0-9]+ ", txt) is None) and (self.get_num_of_tokens(txt) < 20) and (re.match(r"^[ㄱ-ㅣ|가-힣]+ ", txt) is None):
            return False
        else:
            return True

    def split_document(self, document: str) -> List[str]:
        """split the document into chunks with shorter length, this is document-specific.

        Args:
            document (str): document, a long text

        Returns:
            List[str]: a list of chunks
        """
        # remove sub-headers
        document = "".join([p if self.is_paragraph(p) else "\n\n" for p in document.split("\n\n")])

        self.num_of_tokens = self.get_num_of_tokens(document)

        # split documents by newlines
        chunks = [ch for ch in document.split("\n\n") if len(ch) > 0]

        # ensure No. of tokens in each chunk < max context window
        chunks_require_split = list()
        for i, chunk in enumerate(chunks):
            if self.get_num_of_tokens(chunk) > config.CHUNK_SIZE:
                chunks_require_split.append(i)

        text_splitter = CharacterTextSplitter().from_huggingface_tokenizer(
                config.TOKENIZER,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=0, 
        )

        # fine the chunks which need further split
        if len(chunks_require_split) > 0:
            for i in chunks_require_split:
                chunks[i] = text_splitter.split_text(chunks[i])

        # till this step, the chunks' size already less than window size
        chunks = pydash.flatten_deep(chunks)
        #print("no. of chunks:", len(chunks))
        length_max = max([self.get_num_of_tokens(ch) for ch in chunks])
        length_min = min([self.get_num_of_tokens(ch) for ch in chunks])

        logger.info(f"After splitting by paragrah:\ntotal No. of chunks: {len(chunks)}, max length: {length_max}, min length: {length_min}")

        text_splitter = SentenceTransformersTokenTextSplitter( 
            model_name=config.EMBED_PATH,
            )
    
        paragraphs = [s.page_content for s in text_splitter.create_documents(chunks)]

        # get the statistics of setences
        length_max = max([self.get_num_of_tokens(s) for s in paragraphs])
        length_min = min([self.get_num_of_tokens(s) for s in paragraphs])

        logger.info(f"After splitting by sentence:\ntotal No. of paragraphs: {len(paragraphs)}, max length: {length_max}, min length: {length_min}")

        return paragraphs

    def get_embeddings(self, paragraphs: List[str]) -> Dict[str, Dict]:
        """embeddings for each paragraph.
        The API accepts a maximum of 3,072 input tokens and outputs 768-dimensional vector embeddings.
        Use the following parameters for the text embeddings model textembedding-gecko(it belongs to PaLM Model)

        Args:
            paragraphs (List[str]): texts

        Returns:
            Dict[str, Dict]: embeddings
        """

        embedding_dict = dict()
        for idx, para in enumerate(paragraphs):
            sen_embedding = self.embeddings.embed_query(para)

            embedding_dict[str(idx)] = {
                "text": para,
                "embedding": sen_embedding
                }
        return embedding_dict

    def cluster_similar_chunks(self, embedding_dict: Dict[str, Dict], max_cluster: int) -> List:
        """
        cluster chunks into 1 if they share similar semantic meaning
        Args:
            embedding_dict (Dict[str, Dict]): embeddings

        Returns:
            List: list of chunks
        """
        # Get similarity matrix between the embeddings of the sentences' embeddings
        summary_similarity_matrix = np.zeros((len(embedding_dict), len(embedding_dict)))
        summary_similarity_matrix[:] = np.nan

        for row in range(len(embedding_dict)):
            for col in range(row, len(embedding_dict)):
                # Calculate cosine similarity between the two vectors
                similarity = 1 - cosine(embedding_dict[str(row)]["embedding"], embedding_dict[str(col)]["embedding"])
                summary_similarity_matrix[row, col] = similarity
                summary_similarity_matrix[col, row] = similarity

        clusters_out = self.get_clusters(
            summary_similarity_matrix,
            bonus_constant=0.55,
            max_cluster=max_cluster,
            min_size=2)
        clusters = clusters_out['clusters']

        chunks = list()
        for chu_ids in clusters:
            chunk = "\n".join([embedding_dict[str(i)]["text"] for i in chu_ids])
            chunks.append(chunk)

        return chunks

    def get_clusters(self,
                   title_similarity: np.ndarray,
                   bonus_constant: float=0.35,
                   max_cluster: int=512,
                   min_size: int=3) -> Dict[str, List]:
        """calculate if chunks belong to same cluster based on louvain community detection algorithm

        Args:
            title_similarity (np.ndarray): cosine similarity between chunks
            num_clusters (int, optional): number of chunks in the end. Defaults to 8.
            bonus_constant (float, optional): coefficient. Defaults to 0.25.
            min_size (int, optional): minimum size of a chunk. Defaults to 3.

        Returns:
            _type_: _description_
        """
        proximity_bonus_arr = np.zeros_like(title_similarity)
        for row in range(proximity_bonus_arr.shape[0]):
            for col in range(proximity_bonus_arr.shape[1]):
                if row == col:
                    proximity_bonus_arr[row, col] = 0
                else:
                    proximity_bonus_arr[row, col] = 1/(abs(row-col)) * bonus_constant

        title_similarity += proximity_bonus_arr

        title_nx_graph = nx.from_numpy_array(title_similarity)

        # desired_num_clusters = num_clusters
        # Store the accepted partitionings
        n_cluster_accepted = []

        resolution = 0.2
        resolution_step = 0.05
        iterations = 20
        threshold = 1.0e-2
        # Find the resolution that gives the desired number of clusters
        n_cluster = []
        size_max = max_cluster
       
        while size_max > config.CLUSTERING_MAX and resolution < config.RESOLUTION_MAX :
            n_cluster = community.louvain_communities(title_nx_graph, weight = 'weight', resolution = resolution, seed=777, threshold=threshold)
            cluster_size = [len(c) for c in n_cluster]
            size_max = np.max(cluster_size)
            resolution += resolution_step
            
        cluster_size = [len(c) for c in n_cluster]
        sizes_sd = np.std(cluster_size)
        print(n_cluster)
        print("resolution:", resolution)
        lowest_sd_iteration = 0
        # Set lowest sd to inf
        lowest_sd = float('inf')
        #largest_size = 0

        for i in range(iterations):
            n_cluster = community.louvain_communities(title_nx_graph, weight = 'weight', resolution = resolution, seed=999, threshold=threshold)
            # Check SD
            cluster_size = [len(c) for c in n_cluster]
            sizes_sd = np.std(cluster_size)

            n_cluster_accepted.append(n_cluster)

            if sizes_sd < lowest_sd : # and min(cluster_size) < min_size:
                lowest_sd_iteration = i
                lowest_sd = sizes_sd

        # Set the chosen partitioning to be the one with highest modularity
        n_cluster = n_cluster_accepted[lowest_sd_iteration]
        print(n_cluster)
        logger.info(f'Best SD: {lowest_sd}, Best iteration: {lowest_sd_iteration}')

        cluster_id_means = [sum(e)/len(e) for e in n_cluster]
        # Arrange title_clusters in order of cluster_id_means
        n_cluster = [list(c) for _, c in sorted(zip(cluster_id_means, n_cluster), key = lambda pair: pair[0])]
        #print(n_cluster)
        # Create an array denoting which cluster each chunk belongs to
        chunk_cluster = [None] * title_similarity.shape[0]
        for i, c in enumerate(n_cluster):
            for j in c:
                chunk_cluster[j] = i

        return {'chunk_cluster': chunk_cluster,
                'clusters': n_cluster}