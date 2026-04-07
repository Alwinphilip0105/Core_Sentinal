---
language:
- en
tags:
- legal
- business
- psychology
- privacy
size_categories:
- 10K<n<100K
---

# Purpose and Features

The purpose of the model and dataset is to remove personally identifiable information (PII) from text, especially in the context of AI assistants and LLMs.

The model is a fine-tuned version of "Distilled BERT", a smaller and faster version of BERT. It was adapted for the task of token classification based on the largest to our knowledge open-source PII masking dataset, which we are releasing simultaneously. The model size is 62 million parameters. The original encoding of the parameters yields a model size of 268 MB, which is compressed to 43MB after parameter quantization. The models are available in PyTorch, tensorflow, and tensorflow.js

The dataset is composed of ~43’000 observations. Each row starts with a natural language sentence that includes placeholders for PII and could plausibly be written to an AI assistant. The placeholders are then filled in with mocked personal information and tokenized with the BERT tokenizer. We label the tokens that correspond to PII, serving as the ground truth to train our model.

The dataset covers a range of contexts in which PII can appear. The sentences span 54 sensitive data types (~111 token classes), targeting 125 discussion subjects / use cases split across business, psychology and legal fields, and 5 interactions styles (e.g. casual conversation vs formal document).

Key facts:

- Currently 5.6m tokens with 43k PII examples. 
- Scaling to 100k examples
- Human-in-the-loop validated
- Synthetic data generated using proprietary algorithms 
- Adapted from DistilBertForTokenClassification
- Framework PyTorch
- 8 bit quantization

# Performance evaluation

| Test Precision | Test Recall | Test Accuracy |
|:-:|:-:|:-:|
| 0.998636 | 0.998945 | 0.994621 |


Training/Test Set split:

- 4300 Testing Examples (10%)
- 38700 Train Examples

# Community Engagement: 

Newsletter & updates: www.Ai4privacy.com
- Looking for ML engineers, developers, beta-testers, human in the loop validators (all languages)
- Integrations with already existing open source solutions


# Roadmap and Future Development

- Multilingual
- Extended integrations
- Continuously increase the training set
- Further optimisation to the model to reduce size and increase generalisability 
- Next released major update is planned for the 14th of July (subscribe to newsletter for updates)

# Use Cases and Applications

**Chatbots**: Incorporating a PII masking model into chatbot systems can ensure the privacy and security of user conversations by automatically redacting sensitive information such as names, addresses, phone numbers, and email addresses.

**Customer Support Systems**: When interacting with customers through support tickets or live chats, masking PII can help protect sensitive customer data, enabling support agents to handle inquiries without the risk of exposing personal information.

**Email Filtering**: Email providers can utilize a PII masking model to automatically detect and redact PII from incoming and outgoing emails, reducing the chances of accidental disclosure of sensitive information.

**Data Anonymization**: Organizations dealing with large datasets containing PII, such as medical or financial records, can leverage a PII masking model to anonymize the data before sharing it for research, analysis, or collaboration purposes.

**Social Media Platforms**: Integrating PII masking capabilities into social media platforms can help users protect their personal information from unauthorized access, ensuring a safer online environment.

**Content Moderation**: PII masking can assist content moderation systems in automatically detecting and blurring or redacting sensitive information in user-generated content, preventing the accidental sharing of personal details.

**Online Forms**: Web applications that collect user data through online forms, such as registration forms or surveys, can employ a PII masking model to anonymize or mask the collected information in real-time, enhancing privacy and data protection.

**Collaborative Document Editing**: Collaboration platforms and document editing tools can use a PII masking model to automatically mask or redact sensitive information when multiple users are working on shared documents.

**Research and Data Sharing**: Researchers and institutions can leverage a PII masking model to ensure privacy and confidentiality when sharing datasets for collaboration, analysis, or publication purposes, reducing the risk of data breaches or identity theft.

**Content Generation**: Content generation systems, such as article generators or language models, can benefit from PII masking to automatically mask or generate fictional PII when creating sample texts or examples, safeguarding the privacy of individuals.

(...and whatever else your creative mind can think of)

# Support and Maintenance

AI4Privacy is a project affiliated with [AISuisse SA](https://www.aisuisse.com/).