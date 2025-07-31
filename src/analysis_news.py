import streamlit as st
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def find_english_name(russian_name):
    """
    Try to find an English equivalent for a given Russian company name using a predefined mapping
    or by querying online sources (Wikipedia, Investing.com).

    :param russian_name: Company name in Russian
    :return: English company name if found, else None
    """
    name_mapping = {
        "Сбербанк": "Sberbank",
        "Газпром": "Gazprom",
        "Лукойл": "Lukoil",
        "Норникель": "Norilsk Nickel",
        "Роснефть": "Rosneft"
    }

    if russian_name in name_mapping:
        return name_mapping[russian_name]

    try:
        return search_english_name_online(russian_name)
    except:
        return None


def search_english_name_online(russian_name):
    """
    Search for the English equivalent of a Russian company name using Wikipedia and Investing.com.

    :param russian_name: The company name in Russian/Cyrillic characters
    :return: English name if found, None otherwise
    :raises:
        requests.exceptions.RequestException: If there are network-related errors
        Exception: For any other unexpected errors during scraping
    """
    try:
        # Try Wikipedia first
        wiki_url = f"https://en.wikipedia.org/wiki/{russian_name.replace(' ', '_')}"
        try:
            response = requests.get(wiki_url, timeout=10)
            response.raise_for_status()  # Raises HTTPError for bad responses

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.find('h1')
                if title and not "Wikipedia" in title.text:
                    return title.text.strip()

        except requests.exceptions.HTTPError as http_err:
            # Wikipedia page doesn't exist or other HTTP error
            pass
        except requests.exceptions.Timeout:
            # Wikipedia request timed out
            pass
        except requests.exceptions.RequestException as req_err:
            # Other requests-related errors
            pass

        # Fall back to Investing.com
        investing_url = f"https://www.investing.com/search/?q={russian_name}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(investing_url, headers=headers, timeout=10)
            response.raise_for_status()

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                first_result = soup.select_one('.searchSectionMain .js-inner-all-results-quote-item a')
                if first_result:
                    return first_result.text.strip()

        except requests.exceptions.HTTPError as http_err:
            # Investing.com search failed
            pass
        except requests.exceptions.Timeout:
            # Investing.com request timed out
            pass
        except requests.exceptions.RequestException as req_err:
            # Other requests-related errors
            pass

    except Exception as e:
        # Catch any other unexpected errors
        print(f"Unexpected error searching for English name: {str(e)}")
        raise  # Re-raise if you want calling code to handle it

    return None


def show_news_analysis(query, company_name=None, english_name=None):
    """
    Fetch and display the latest news about a company or topic, along with sentiment analysis
    using both TextBlob and transformer-based models.

    :param query: The search term, e.g., ticker or company keyword
    :param company_name: Optional official name of the company
    :param english_name: Optional English name (used to improve relevance)
    :return: Dictionary containing article info, sentiment scores, model name, and language
    """
    st.subheader(f"📢 Latest News and Sentiment for {company_name or query}")

    # Display search terms being used
    search_terms = [query]
    if company_name:
        search_terms.append(company_name)
    if english_name:
        search_terms.append(english_name)

    st.markdown(f"**Search terms:** {', '.join(search_terms)}")

    # Language selection with smart default
    if english_name and not (query.isalpha() and query.isupper()):
        default_lang = "en"
    else:
        default_lang = "ru"

    language = st.selectbox(
        "Select news language",
        options=["en", "ru"],
        index=0 if default_lang == "en" else 1,
        key=f"news_lang_{query}"
    )

    def fetch_news(search_terms, page_size=5):
        """
        Fetch news articles using the NewsAPI based on a list of search terms.

        :param search_terms: List of strings to search news for
        :param page_size: Number of articles to return
        :return: List of news article dictionaries
        """
        articles = []
        for term in search_terms:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": term,
                "apiKey": NEWS_API_KEY,
                "language": language,
                "sortBy": "relevance",
                "pageSize": page_size,
            }
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                articles.extend(response.json().get("articles", []))
            except Exception as e:
                st.warning(f"Couldn't fetch news for '{term}': {str(e)}")
        return articles[:page_size]  # Return only top results

    # Get articles using all search terms
    articles = fetch_news(search_terms)

    if not articles:
        st.warning(f"No news found for: {', '.join(search_terms)}")
        return

    st.markdown("### 📰 Latest News")
    news_texts = []
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        combined_text = f"{title}. {description}".strip()
        news_texts.append((title, description, combined_text))

        st.markdown(f"#### [{title}]({article.get('url')})")
        st.write(description)
        st.markdown("---")

    # TextBlob Analysis
    st.markdown("### 🧪 Sentiment Analysis — TextBlob")
    blob_scores = [TextBlob(text).sentiment.polarity for _, _, text in news_texts]
    avg_blob = sum(blob_scores) / len(blob_scores)

    st.write(f"**Average Sentiment Score:** {avg_blob:.2f}")
    if avg_blob > 0.1:
        st.success("Overall Positive Sentiment")
    elif avg_blob < -0.1:
        st.error("Overall Negative Sentiment")
    else:
        st.info("Overall Neutral Sentiment")

    # Transformers Analysis
    st.markdown("### 🤖 Advanced Sentiment Analysis")
    model_name = "blanchefort/rubert-base-cased-sentiment" if language == "ru" else "nlptown/bert-base-multilingual-uncased-sentiment"
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)

    tf_scores = []
    for _, _, text in news_texts:
        result = sentiment_analyzer(text[:512])[0]
        if language == "ru":
            polarity = {"positive": 1, "neutral": 0, "negative": -1}.get(result["label"].lower(), 0)
        else:
            stars = int(result["label"][0])
            polarity = 1 if stars > 3 else (-1 if stars < 3 else 0)
        tf_scores.append(polarity)

    avg_tf = sum(tf_scores) / len(tf_scores)
    st.write(f"**Average Sentiment Score:** {avg_tf:.2f}")
    if avg_tf > 0.1:
        st.success(f"{model_name}: Positive Sentiment")
    elif avg_tf < -0.1:
        st.error(f"{model_name}: Negative Sentiment")
    else:
        st.info(f"{model_name}: Neutral Sentiment")

    news_results = {
        "articles": [],
        "textblob_score": avg_blob,
        "transformers_score": avg_tf,
        "transformers_model": model_name,
        "language": language
    }

    for article in articles:
        news_results["articles"].append({
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "url": article.get("url", "")
        })

    return news_results
