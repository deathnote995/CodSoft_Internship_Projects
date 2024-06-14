{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPbyyATo57msKuu36//pKOg",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/deathnote995/CodSoft_Internship_Projects/blob/main/MOVIE_GENRE.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ylg-R_GgIY8i",
        "outputId": "7ddaafaf-d489-4d1c-f763-99356dbe2e8e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Loading Train Data: 100%|██████████| 50/50 [00:00<00:00, 329.46it/s]\n",
            "Vectorizing Training Data: 100%|██████████| 50/50 [00:02<00:00, 24.61it/s]\n",
            "Training Model: 100%|██████████| 50/50 [00:00<00:00, 141.22it/s]\n",
            "Loading Test Data: 100%|██████████| 50/50 [00:00<00:00, 1556.34it/s]\n",
            "Vectorizing Test Data: 100%|██████████| 50/50 [00:00<00:00, 99.26it/s]\n",
            "Predicting on Test Data: 100%|██████████| 50/50 [00:00<00:00, 825.09it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model evaluation results and metrics have been saved to 'model_evaluation.txt'.\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.multioutput import MultiOutputClassifier\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.preprocessing import MultiLabelBinarizer\n",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
        "from tqdm import tqdm\n",
        "\n",
        "# List of Genres learnt from the Training Data\n",
        "genre_list = [ 'action', 'adult', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'family', 'fantasy', 'game-show', 'history', 'horror', 'music', 'musical', 'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show', 'thriller', 'war', 'western' ]\n",
        "\n",
        "# Define a fallback genre for movies which the model finds very hard to predict\n",
        "fallback_genre = 'Unknown'\n",
        "\n",
        "# Load the Training dataset from train_data.txt\n",
        "try:\n",
        "    with tqdm(total=50, desc=\"Loading Train Data\") as pbar:\n",
        "        train_data = pd.read_csv('/train_data.txt', sep=':::', header=None, names=['SerialNumber', 'MOVIE_NAME', 'GENRE', 'MOVIE_PLOT'], engine='python')\n",
        "        pbar.update(50)\n",
        "except Exception as e:\n",
        "    print(f\"Error loading train_data: {e}\")\n",
        "    raise\n",
        "\n",
        "# Data preprocessing for training data\n",
        "X_train = train_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())\n",
        "genre_labels = [genre.split(', ') for genre in train_data['GENRE']]\n",
        "mlb = MultiLabelBinarizer()\n",
        "y_train = mlb.fit_transform(genre_labels)\n",
        "\n",
        "# TF-IDF Vectorization\n",
        "tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features\n",
        "\n",
        "# Fit and transform the training data with progress bar\n",
        "with tqdm(total=50, desc=\"Vectorizing Training Data\") as pbar:\n",
        "    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)\n",
        "    pbar.update(50)\n",
        "\n",
        "# Train a MultiOutput Naive Bayes classifier using the training data\n",
        "with tqdm(total=50, desc=\"Training Model\") as pbar:\n",
        "    naive_bayes = MultinomialNB()\n",
        "    multi_output_classifier = MultiOutputClassifier(naive_bayes)\n",
        "    multi_output_classifier.fit(X_train_tfidf, y_train)\n",
        "    pbar.update(50)\n",
        "\n",
        "# Load your test dataset from test_data.txt\n",
        "try:\n",
        "    with tqdm(total=50, desc=\"Loading Test Data\") as pbar:\n",
        "        test_data = pd.read_csv('/test_data.txt', sep=':::', header=None, names=['SerialNumber', 'MOVIE_NAME', 'MOVIE_PLOT'], engine='python')\n",
        "        pbar.update(50)\n",
        "except Exception as e:\n",
        "    print(f\"Error loading test_data: {e}\")\n",
        "    raise\n",
        "\n",
        "# Data preprocessing for test data\n",
        "X_test = test_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())\n",
        "\n",
        "# Transform the test data with progress bar\n",
        "with tqdm(total=50, desc=\"Vectorizing Test Data\") as pbar:\n",
        "    X_test_tfidf = tfidf_vectorizer.transform(X_test)\n",
        "    pbar.update(50)\n",
        "\n",
        "# Predict genres on the test data\n",
        "with tqdm(total=50, desc=\"Predicting on Test Data\") as pbar:\n",
        "    y_pred = multi_output_classifier.predict(X_test_tfidf)\n",
        "    pbar.update(50)\n",
        "\n",
        "# Create a DataFrame for test data with movie names and predicted genres\n",
        "test_movie_names = test_data['MOVIE_NAME']\n",
        "predicted_genres = mlb.inverse_transform(y_pred)\n",
        "test_results = pd.DataFrame({'MOVIE_NAME': test_movie_names, 'PREDICTED_GENRES': predicted_genres})\n",
        "\n",
        "# Replace empty unpredicted genres with the fallback genre\n",
        "test_results['PREDICTED_GENRES'] = test_results['PREDICTED_GENRES'].apply(lambda genres: [fallback_genre] if len(genres) == 0 else genres)\n",
        "\n",
        "# Write the results to an output text file with proper formatting\n",
        "with open(\"/model_evaluation.txt\", \"w\", encoding=\"utf-8\") as output_file:\n",
        "    for _, row in test_results.iterrows():\n",
        "        movie_name = row['MOVIE_NAME']\n",
        "        genre_str = ', '.join(row['PREDICTED_GENRES'])\n",
        "        output_file.write(f\"{movie_name} ::: {genre_str}\\n\")\n",
        "\n",
        "\n",
        "# Calculate evaluation metrics using training labels (as a proxy)\n",
        "y_train_pred = multi_output_classifier.predict(X_train_tfidf)\n",
        "\n",
        "# Calculate evaluation metrics\n",
        "accuracy = accuracy_score(y_train, y_train_pred)\n",
        "precision = precision_score(y_train, y_train_pred, average='micro')\n",
        "recall = recall_score(y_train, y_train_pred, average='micro')\n",
        "f1 = f1_score(y_train, y_train_pred, average='micro')\n",
        "\n",
        "# Append the evaluation metrics to the output file\n",
        "with open(\"model_evaluation.txt\", \"a\", encoding=\"utf-8\") as output_file:\n",
        "    output_file.write(\"\\n\\nModel Evaluation Metrics:\\n\")\n",
        "    output_file.write(f\"Accuracy: {accuracy * 100:.2f}%\\n\")\n",
        "    output_file.write(f\"Precision: {precision:.2f}\\n\")\n",
        "    output_file.write(f\"Recall: {recall:.2f}\\n\")\n",
        "    output_file.write(f\"F1-score: {f1:.2f}\\n\")\n",
        "\n",
        "\n",
        "print(\"Model evaluation results and metrics have been saved to 'model_evaluation.txt'.\")"
      ]
    }
  ]
}