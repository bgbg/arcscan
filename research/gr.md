Grok: https://grok.com/c/b6cc41bc-7ae3-4996-8259-58762cba4871?rid=0906aa94-377c-441f-9218-f50c511ee38c



### Key Insights on Emotional Arc Analysis in Political Speeches
- Research suggests that analyses similar to the 2016 study on story arcs have appeared in the last decade, though they often adapt the approach rather than replicate it exactly, with mixed results on identifying dominant shapes in political contexts.
- Evidence leans toward qualitative or modified quantitative methods for political narratives, such as identifying recurring emotional stages like love-fear-anger in populist rhetoric, or clustering sentiment trajectories in public speeches, but controversy exists around whether fiction-based arc models fully transfer to real-world political discourse.
- It seems likely that no exact replication dominates political speeches, as studies highlight variability in arcs due to context, but applications to non-fictional narratives, including politics, show promise for understanding audience impact.

#### Recent Similar Analyses
Several studies since 2016 have explored emotional arcs or similar sentiment trajectories in political or speech-based contexts, building on the foundational work by Reagan et al. (2016). For instance, a 2021 analysis applied sentiment scoring, singular value decomposition (SVD), and k-means clustering to Brazilian political Facebook posts from the 2018 election, finding that arcs did not cluster into the six shapes from fiction but appeared more random, raising questions about domain transfer. A 2017 qualitative examination of populist campaigns (e.g., Trump, Farage, Le Pen) identified a common three-stage arc of love for the homeland, fear of outsiders, and anger at elites in speeches, emphasizing emotional mobilization without quantitative clustering. Additionally, a study on TED talks used IBM Watson for emotion extraction, DBSCAN clustering for trajectory shapes (e.g., flat, increasing), and linked them to audience ratings, showing flat joy arcs correlated with perceptions of being "longwinded." A 2025 article extended arc analysis to non-fiction, including political briefings like Trump's and Cuomo's COVID-19 speeches, noting valence fluctuations tied to events. These suggest the approach is viable but requires adaptation for politics.

#### Proposed Paper Introduction
**Introduction**

Political speeches represent a cornerstone of democratic discourse, serving not only as vehicles for policy articulation but also as instruments of emotional persuasion and ideological framing. Over the past decade, natural language processing (NLP) has revolutionized the analysis of such speeches, enabling researchers to uncover patterns in sentiment, rhetoric, and structure that were previously inaccessible through manual methods alone. State-of-the-art NLP techniques for political speeches include sentiment analysis, which employs lexicon-based tools like VADER or advanced models like BERT to detect emotional tones; topic modeling via latent Dirichlet allocation (LDA) or neural variants to identify thematic clusters; stance detection to classify positions on issues using transformer-based classifiers; argument mining to extract premises and claims with models like Longformer; and persuasion assessment, which integrates rhetorical feature extraction (e.g., hyperbole, concreteness) with deep learning for propaganda and fallacy detection. Recent advances leverage large language models (LLMs) like RoBERTa for multi-granular analysis of UN speeches or U.S. presidential addresses, revealing trends in emotional intensity and ideological shifts. Hybrid approaches combining ontologies for structured knowledge representation with LLMs further enhance interpretability, as seen in frameworks for diachronic semantic shifts in terms like "immigrant" across political corpora.

Building on this foundation, the concept of emotional arcs—time-series representations of sentiment fluctuations—inspired by narrative theory, offers a novel lens for dissecting political speeches. Originating from computational literary analysis, where Reagan et al. (2016) used hedonometer scores and k-means clustering to identify six dominant shapes (e.g., "man in a hole," "rags to riches") in fiction, this approach has been extended to non-fictional domains. Clustering techniques, such as DBSCAN for trajectory grouping or SVD for dimensionality reduction, allow for the identification of recurrent patterns, linking arcs to outcomes like audience engagement or persuasion efficacy. In political contexts, arcs manifest as deliberate emotional manipulations, such as rising joy in inspirational segments or peaking anger in calls to action, as observed in populist rhetoric.

Despite these advancements, several open questions persist. How do emotional arc shapes vary across political ideologies or cultural contexts, and do they predict electoral success? Can NLP models integrate multimodal data (e.g., intonation, gestures) to refine arc detection, addressing current limitations in text-only analysis? Ethical concerns around bias in training data and the potential for arc manipulation in synthetic speeches generated by LLMs remain underexplored. Furthermore, the generalizability of fiction-derived clustering to dynamic, real-time political discourse—where arcs may be fragmented or context-dependent—warrants deeper investigation. This study applies emotional arc analysis to a corpus of politicians' speeches, clustering sentiment trajectories to uncover common shapes and their implications for persuasion, contributing to bridging these gaps.

---
Political speeches have long been recognized as powerful tools for shaping public opinion, mobilizing support, and framing societal issues, with their emotional underpinnings playing a central role in effectiveness. The integration of natural language processing (NLP) has transformed how scholars dissect these texts, moving from qualitative interpretations to data-driven insights. This comprehensive review draws on recent literature to outline the state of the art in NLP for political speeches, the application of emotional arc concepts with clustering methods, and persisting open questions, setting the stage for an analysis of common emotional shapes in politicians' speeches.

#### State of the Art in NLP Analysis of Political Speeches
NLP techniques have evolved significantly, particularly with the advent of deep learning, enabling nuanced examination of political discourse. Key methods include:

- **Sentiment Analysis**: Tools like VADER (Valence Aware Dictionary and sEntiment Reasoner) and transformer models such as BERT extract emotional tones at sentence or document levels. For instance, analyses of U.S. State of the Union addresses reveal long-term trends in positivity and rhetorical tone, with sentiment responding to economic conditions. In UN speeches on global conflicts, sentiment polarity highlights diplomatic stances, achieving high accuracy with fine-tuned LLMs.

- **Topic Modeling and Stance Detection**: LDA and neural topic models identify thematic clusters, while stance detection classifiers (e.g., using cosine similarity of embeddings) pinpoint positions on issues like migration. Studies on party manifestos use these to compare ideologies, with BERT variants reaching F1 scores of 0.85+.

- **Argument Mining and Persuasion Assessment**: Computational argumentation extracts structures (premises, claims) from debates, employing models like RoBERTa for fallacy detection (e.g., ad hominem). Persuasion studies analyze rhetorical figures via LIWC dictionaries, with hybrid ontology-LLM approaches improving explainability in propaganda identification.

- **Diachronic and Multimodal Advances**: Embeddings like Word2Vec track semantic shifts in terms (e.g., "refugee" over decades), while multimodal methods incorporate audio-visual cues from speeches. Recent scoping reviews note NLP's role in polarization detection, with over 90% of studies focusing on English but expanding to multilingual datasets.

These techniques are applied to diverse corpora, from congressional records to social media echoes of speeches, revealing patterns like increased moralized language in national vs. local politics.

| Technique | Example Applications in Political Speeches | Key Models/Tools | Performance Metrics (Recent Studies) |
|-----------|-------------------------------------------|------------------|-------------------------------------|
| Sentiment Analysis | Trend detection in State of the Union; UN speech polarity | VADER, BERT, RoBERTa | F1: 0.76-0.85; AUC: up to 0.88 |
| Topic Modeling | Manifesto comparison; framing in debates | LDA, Neural TM | Coherence scores: 0.5-0.7 |
| Stance Detection | Position classification on policies | Cosine embeddings, Longformer | Accuracy: 80-90% |
| Argument Mining | Fallacy/propaganda extraction | Ontology-hybrid LLMs | F1: 0.84 for fallacies |
| Diachronic Shifts | Evolution of terms like "immigrant" | Word2Vec, BERT | Cosine distance shifts: 0.2-0.4 over decades |

#### Emotional Arc Shapes and Clustering
Inspired by Kurt Vonnegut's narrative theories, emotional arcs plot sentiment over time, often using happiness scores from lexicons. The seminal 2016 study clustered 4,803 fiction stories into six shapes via k-means on SVD-reduced trajectories, identifying patterns like "rags to riches" (rising) or "tragedy" (falling). Extensions include EmotionArcs (2024) for 9,000 literary texts, using hierarchical clustering to find six clusters. In non-fiction, arcs challenge plot-centric views, revealing latent structures in memoirs or social media aggregates.

For speeches, adaptations use tools like IBM Watson for multi-emotion scores (joy, anger), applying smoothing filters and interpolation to canonical lengths. Clustering via DBSCAN yields shapes like flat (correlating with negative ratings) or increasing (positive engagement). In politics, a 2021 study on Brazilian narratives found arcs via VADER and k-means but noted poor fit to fiction shapes. Populist speeches often follow a love-fear-anger arc, qualitatively analyzed across campaigns. ArXiv works mention arcs in political discourse for sentiment challenges and leader visualizations.

| Arc Shape Example | Description | Political Application | Clustering Method |
|-------------------|-------------|-----------------------|-------------------|
| Rags to Riches | Rising sentiment | Inspirational rallies building hope | k-means |
| Man in a Hole | Fall then rise | Crisis framing followed by resolution | SVD + k-means |
| Flat | Stable emotion | Descriptive policy sections; linked to "longwinded" ratings | DBSCAN |
| Love-Fear-Anger | Sequential stages | Populist mobilization in campaigns | Qualitative/trajectory analysis |
| Increasing Joy | Gradual build-up | Motivational closings | Interpolation + clustering |
| Peaked | Mid-high, ends low | Warning speeches with unresolved tension | Neighbor-based |

#### Open Questions and Challenges
While NLP has advanced, gaps remain. Transferability of fiction arcs to politics is debated, as real discourse may yield random clusters due to variability. Explainability in black-box models hinders trust, especially for ethical issues like bias amplification in polarized data. Multilingual and multimodal integration is limited, with most studies English-focused; how do arcs interact with visuals or intonation? Open questions include: Do arc shapes predict voter behavior or polarization? How can generative LLMs simulate arcs ethically? What cultural variations exist, and how to address synthetic speech detection? Future work should prioritize hybrid models and diverse corpora to tackle these.

This analysis extends prior work by applying arc clustering to politicians' speeches, potentially revealing ideology-specific shapes and their persuasive roles.

### Key Citations
- https://www.researchgate.net/publication/348677113_The_emotional_arcs_of_political_narratives
- https://journals.gmu.edu/NandC/article/download/1954/1378
- https://www.cs.rochester.edu/hci/pubs/pdfs/TED.pdf
- https://culturalanalytics.org/article/143671-beyond-plot-how-sentiment-analysis-reshapes-our-understanding-of-narrative-structure
- https://hybridsproject.eu/wp-content/uploads/2025/02/D2.1-Technical-report-on-the-state-of-the-art-of-NLP_v1.pdf
- https://www.diva-portal.org/smash/get/diva2:1980354/FULLTEXT02
- https://www.frontiersin.org/journals/political-science/articles/10.3389/fpos.2025.1546822/full
- https://cdanfort.w3.uvm.edu/research/2016-reagan-epj.pdf
- https://aclanthology.org/2024.latechclfl-1.7.pdf
- https://arxiv.org/pdf/2301.09912
- https://arxiv.org/abs/1601.03313
- https://arxiv.org/pdf/2505.16274
- https://academic.oup.com/ej/article/132/643/1037/6490125
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9762668/
- https://www.sbp-brims.org/2022/papers/working-papers/2022_SBP-BRiMS_Final_Paper_PDF_3597.pdf
- https://elliottash.com/wp-content/uploads/2021/08/Emotions_in_Politics_Economic_Journal_CA.pdf