# The Recognition for the News Topic Written by a Human or a Machine

* ### Introduction
  >The project is aim to recognize the news topic written by humans or generated by the machine. The news in the dataset provided by Kaggle were obtained from HuffPost. The dataset contains the links of each news source, so we crawled the topic written by human via the links, labeled as 0. In the mean time, we applied T5 and Transformer Summerization methods to generate the topic, labeled as 1. Then, we conducted a classification task with supervised learning to recognize who produced the topic. Finally, F1-score and false negative were adopted to analyze the results.

* ### Technique
  * T5 Fine-tune
    >We conducted the topic generation task with T5 model, which converts all downstream task as a text-to-text format. This model applied the pre-trained model, t5-base. Then, we fine-tuned the model by ourselves on the generation task.
 
  * T5 Michau
    >Different from above model, this model used the pre-trained model, Michau, which has been trained on a collection of 500k articles with headings.
  
  * Transformer Summarization
    >Headliner is a sequence modeling library for generating headlines from Welt news articles. It was produced to an API. Therefore, we can apply the method by importing their library.
 
  * BERT Classification
    >We adopted the bert-large-uncased as the pre-trained model of BERT classification task. The topics generated by machine were labeled to 1, and written by humans were labeled to 0.

* ### Result
  * F1-Score
    * T5 Fine-tune: 63.99 %
    * T5 Michau: 85.67 %
    * Headliner: 87.99 %
  * NF
    * T5 Fine-tune: 35.50 %
    * T5 Michau: 10.40 %
    * Headliner: 13.20 %
    
* ### Supplement
  * Document
    >The document file contains the presentation and final report for DL course. If wondering the project in details in Chinese, you can check the files there.
  
