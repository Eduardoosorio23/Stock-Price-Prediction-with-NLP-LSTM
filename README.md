# Stock Price Prediction With sentiment analysis
By: Eduardo Eduardoosorio23

## Objective:
The purpose of this study was to compare stock prices with sentiment towards said stock and see if that affects its future price. I will be using NLP for sentiment analysis and a Long Short Term Memory Recurrent Neural Network for future stock price prediction. This will give individual or businesses insight on whether they should buy or sell a stock based on how the public feels.

##The data:
The dataset was synthesized using **Psaw** for reddit post and **Twint** for twitter posts. The ticker symbol used to train this model was **TSLA(Tesla)** and I was able to extract 7k+ reddit/twitter posts. I used **SpaCy** which is a **Natural Language Processing** tool to essentially teach the model how to read textual data. I also used **SpaCy's TextBlob pipeline component** to find post **sentiment** and **subjectivity**. **Sentiment** is measured from 1 to -1, 1 being positive -1 being negative. **Subjectivity** is measured from 1 to 0, 1 being highly opinionated and 0 being very factual.

Here is an example of how the model reads a post:
![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Capstone/main/Data/Pictures/displaCy%20PoS%20Tagging2.png?token=APSW5OH4PFACGX4SGT6PE53ASL3KW)

## EDA:
During the initial EDA phase the two major insights that stood out were:
- The **less subjective** a tweet the more **neutral** as far as **sentiment**.
- When a spike in **sentiment** happens we can expect a **major correction** in the near future.

![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Capstone/main/Data/Pictures/subjectivity%20vs%20sentiment.png?token=APSW5OA6WLEPG6M6MLHKR43ASL2WM)

![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Capstone/main/Data/Pictures/Closing%20Price%20and%20Sentiment%20history.png?token=APSW5OAUW6NJ5KYIIX7GV4LASL2YE)

## Base model:
In order to have the data in a format usable to this model, I had to **compress** the number of **sentiment** and **subjectivity** values into 1 value per day. The post data had 7k+ post spread out through 24 months. I used **groupby()** and **mean()** to group all the post by dates and return the average **sentiment**/**subjectivity** values. Next I made a **target** column by shifting the closing prices so tomorrows closing price would equal todays target. The architecture for the **base model** included a **LSTM** with 2 **LSTM** layers, 1 **Dense** layer, **batch size** of 1 and 100 **Epochs**. The ** Test Mean Absolute Percentage Error was **15.3%** with an **RMSE** of **141.55**. That means the model was off by an average of **$141.55**.

![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Capstone/main/Data/Pictures/Base%20model%20results.png?token=APSW5OHKPGJN7VUHZKDYABLASL25A)

![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Capstone/main/Data/Pictures/Base%20Model%20graph.png?token=APSW5OECJ5QNFDQPA7DOUC3ASL27S)


## Final Model:
For the final model, I tried different combinations of layers, including:
- **Dropout** layers.
- **L2 regularization**.
- **Early stopping**.
- Increasing/decreasing the number of **epochs**.
- Increasing the **batch size**.

The architecture for the final model was 2 **LSTM** layers, 3 **Dense** layers, **batch size** of 20 and 100 **epochs**. Compared to the Test Model, I was able to lower the **Test Mean Absolute Percentage Error** by **250%**. The **Training Mean Absolute Percentage Error** is **5.68%** and the RMSE is **48.41**. The final Model is off by an average of **$48.41** on a stock that's currently trading at **$671**.

![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Capstone/main/Data/Pictures/Final%20model%20results.png?token=APSW5OCS6DPN4DF2DCFELYDASL3BY)

![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Capstone/main/Data/Pictures/Final%20Model%20graph.png?token=APSW5OFSQKFEI52PQNLIVPTASL3EA)

## Summary /Conclusion
- LSTM’s work very well on forecasting time series data.
- Sentiment can signal if there's going to be a significant decrease or increase in the near future.
