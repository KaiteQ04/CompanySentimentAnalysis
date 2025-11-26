# Company-Sentiment-Analysis
This project is designed to systematically monitor and analyze public sentiment toward listed companies by leveraging multi-source unstructured data, including financial news, corporate annual reports, and stock forum comments.

This project constructed corporate sentiment factors and apply them to ETF investment strategies. The workflow is as follows:

Data Collection and Cleaning: Crawled news articles, forum comments, and annual reports of listed companies from 2014 to 2024; Standardized data, removed outliers, and performed feature selection and dimensionality reduction.

LLM & Sentiment Factor Construction: Ensured labeling consistency through collaborative manual annotation to produce high-quality labels for LLM training; Fine-tuned Qwen LLM for financial text understanding; Annotated text sentiment using a financial sentiment dictionary combining general, professional, and colloquial terms, together with NLP techniques; Generated "Sentiment Index" from news articles to quantify market sentiment.

Technical and Integrated Factor Construction: Generated 158 technical factors from historical A-share price-volume data; Combined sentiment and technical factors to form 171 integrated factors; Fitted asset returns using a LightGBM model to generate an integrated factor reflecting market crowding for dynamic market evaluation.

Factor Evaluation and ETF Strategy Backtesting: Evaluated factor effectiveness using IC, Rank IC, and IR metrics; Designed a TopKStrategy ETF rotation strategy (long top K% assets, short bottom K%); Backtesting results show a cumulative return of 14.95% and a Sharpe Ratio of 1.47, with controlled risk.
