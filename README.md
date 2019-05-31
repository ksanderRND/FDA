# FDA
This is a practical assignment for Fuzzy Data Analisys course at LUT

Practical assignment: Classification of bank marketing data set.

Student: Lukoshkin Aleksandr.
   
Professor: Pasi Luukka.

This work is a practical assignment for the Fuzzy Data Analysis course. The dataset is given by:
S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. 
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls.The dataset consist of 16 attributes:
   1. - age (numeric)
   2. - job : type of job (categorical: "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services") 
   3. - marital : marital status (categorical: "married", "divorced", "single")
   4. - education (categorical: "unknown", "secondary", "primary", "tertiary")
   5. - default: has credit in default? (binary: "yes", "no")
   6. - balance: average yearly balance, in euros (numeric) 
   7. - housing: has housing loan? (binary: "yes", "no")
   8. - loan: has personal loan? (binary: "yes", "no")
   
 Related with the last contact of the current campaign:

   9. - contact: contact communication type (categorical: "unknown", "telephone", "cellular") 
   10. - day: last contact day of the month (numeric)
   11. - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
   12. - duration: last contact duration, in seconds (numeric)
   
 Other attributes:
 
   13. - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
   14. - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
   15. - previous: number of contacts performed before this campaign and for this client (numeric)
   16. - poutcome: outcome of the previous marketing campaign (categorical: "unknown", "other", "failure", "success")
   
 And as desired outcome:
 
   17. - y - has the client subscribed a term deposit? (binary: "yes", "no")

Classification task solved by using (handmade algorithms in Matlab):

   a) Fully-connected neural network;
   b) Similarity classifier;
 
Behind that, in case to measure importance of the features there were implemented similarity measure and fuzzy entropy measure.
 
The results:

Without feature removing:

a)For nueral network:
  - Number of misclassified test samples: 0009 out of 9043
  - Accuracy:   0.999005

b)For similarity classifier
  - Accuracy:   0.961738
  - Number of misclassified test samples: 0346 out of 9043

With feature removing:

a)For nueral network:
  - Number of misclassified test samples: 0002 out of 9043
  - Accuracy:   0.999779

b)For similarity classifier
  - Accuracy:   0.996351
  - Number of misclassified test samples: 0033 out of 9043
