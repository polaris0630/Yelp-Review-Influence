# Yelp-Review-Influence
Investigate how prior Yelp reviews influence future reviews


This project is about discovering the influence of prior reviews on future reviews on Yelp.

Procedures:

1) Raw Yelp datasets (from Yelp data challenge year 2019&2020) were used and restaurants with more than 500 reviews were selected. All selected businesses were matched with their corresponding reviews, as well as the users of the reviews, respectively.

2) Raw features associated with review, text, and user were extracted, including:

   a. Review features - review star ratings, number of votes, etc.

   b. Text features - TFIDF, mean_probability of unigrams for each review, review length, sentiment features (polarity and subjectivity).

   c. User features - number of friends, number of followers, yelping_since, elite count, votes received, etc.

   d. Interaction between star ratings and text features (sentiment, tfidf, and mean_probabilty) were calculated.

   17 features in total were selected eventually, considering the correlation.

3. Created Hawkes variables: All raw features were processed through the Hawkes Point Process model to aggregate the influence received by prior reviews, where 5 different values were implemented on the decay parameters of the model to control the decay speed of the dissemination process.

   17*5 = 85 number of Hawkes variables were created.

4. Hawkes vars were inputted into the Logistic Regression model to predict the star ratings of the current review, where the feature importance will tell whether a variable (feature) is statistically significant, thus, has a significant influence on future reviews.

5. Simulation: generated review star ratings using Multinomial Logistic Regression model along with the distribution of the review star ratings of each restaurant, assigned the generated data with shuffled reviews as the simulated data, and re-inputted into the Logistic Regression model, to verify the findings from the real data.

6. Velidation: extracted business features (price, region, business type etc.) and predicted whether the reviews of a business contains at least one significant var (feature) on affecting the future reviews.
