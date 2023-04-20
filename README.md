# bootcamp_capstone


The link to the app is found here: https://pcammall-bootcamp-c-sentiment-classifier-streamlit-input-3rxojp.streamlit.app/
This is the user-friendly screen that can take a Financial headline. Manually entering text works, but it is more useful to directly copy and paste the headline. It will then process and return the sentiment. It does show both a number and text labels for the sentiment. This classifier does have some limited capability in that it focuses on financial headlines, thus "ice cream party" may not be classified correctly, while "stock market crash" will. In the case of Neutral sentiment ("market trades sideways"), this will be counted as a Positive sentiment.



The sentiment_classifier_streamlit_input.py file contains the code to run this via Streamlit.io. Please follow their instructions to install Streamlit on your system before running this file. The code contained within this file is minimal, as it uses a saved model. The training for that model has already occured. Perioditcally the saved model will be updated and improved, which will result in an update to the app that may disrupt usage of it. 

Note, that since this was deployed with Streamlit, it is subject to the limitations that Streamlit has. That is, if modifications are made to the code, they must be compatible with Streamlit. This may require extra work, for example, downloading more items for NLTK, or avoiding PyWin32 (Streamlit deploys to Linux environment). Note that while PyWin32 is not relevant for this file, the NLTK items are, to properly process the input text.