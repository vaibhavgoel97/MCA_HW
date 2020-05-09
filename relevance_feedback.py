import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def relevance_feedback(vec_docs, vec_queries, sim, gt, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    for count in range(3):
        alpha = 1
        beta = 0.2
        new_queries = []
        for i in range(vec_queries.shape[0]):
            dR = np.zeros((1, 10625))
            dNR = np.zeros((1, 10625))
            rel_docs = np.argsort(-sim[:, i])[:n]
            # print('rel', rel_docs)
            gt_rel = []
            for j in gt:
                if j[0] == i + 1:
                    gt_rel.append(j[1])
            # print('gt', gt_rel)
            counter = 0
            for j in rel_docs:
                # print("REL:    ", j)
                if j in gt_rel:
                    counter +=1
                    # print("REL:     ", j)
                    dR += vec_docs[j] 
                else:
                    dNR += vec_docs[j]
            # print(counter)      
            vec_queries[i] = vec_queries[i] + ((alpha*dR) - (beta*dNR))         
    rf_sim = cosine_similarity(vec_docs, vec_queries)
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, gt, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    for count in range(3):
        alpha = 1
        beta = 0.2
        new_queries = []
        for i in range(vec_queries.shape[0]):
            dR = np.zeros((1, 10625))
            dNR = np.zeros((1, 10625))
            rel_docs = np.argsort(-sim[:, i])[:n]
            # print('rel', rel_docs)
            gt_rel = []
            for j in gt:
                if j[0] == i + 1:
                    gt_rel.append(j[1])
            # print('gt', gt_rel)
            counter = 0
            top10 = np.zeros((1, 10625))
            if len(gt) >= 10:
                top10 = getTop10Terms(gt_rel[:10], vec_docs)
            else:
                top10 = getTop10Terms(gt_rel[:10], vec_docs)
            # print(np.nonzero(newer))
            # top10 = top10 + newer

              
            # print(counter)      
            vec_queries[i] = vec_queries[i] + top10     
    rf_sim = cosine_similarity(vec_docs, vec_queries)
    return rf_sim

def getTop10Terms(gt_rel, vec_docs):
    # print('gt', len(gt_rel))
    index = []
    value = []
    for i in gt_rel:
        docs = vec_docs[i-1].toarray()[0]
        # print(len(docs))
        for j in range(len(docs)):
            value.append(docs[j])
            index.append(j)
   
    value, index = zip(*sorted(zip(value, index), reverse=True))
    # print(len(index), len(value))   
    top10 = np.zeros((1, 10625))
    for j in range(10):
        temp = np.zeros((1, 10625))
        temp[0][index[j]] = value[j]
        top10 = top10 + temp

    return top10