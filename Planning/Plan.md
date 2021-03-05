# Nueral Net COVID-19 Predictior and Data Display:

## Program Flow:
### Plan:
- [Draw IO Plan Link](https://drive.google.com/drive/folders/1XMwFZWw3onM0upoKn9mJDuUc5xnZPpnF) - Also found in DRAWIOPLAN.drawio in same dir (Must have draw.io extension to view)
- Scraper(Scrapy) -> Model(Tensor Flow) -> Website(Django)
### Project Refernce(Includes Data Display but not prediction model):
- [John Hopkins Covid Tracker](https://coronavirus.jhu.edu/map.html)

## Scraper:
### Overview:
- Using Scrapy
- Create a Spider to crawl through and scrape one or more websites to gather the following info:
1. Total cases each day
- Store this infromation in a file named Data.JSON using pythons JSON read and write functions
6. NEED TO FIND A WAY TO ORGANIZE DATA BEFORE STORING I.E PANDAS OR NUMPY
### Scraper Targets:
1. [Google Analytics Stats](https://www.google.com/search?q=putnam+valley+covid+stats&rlz=1C1RXQR_enUS933US933&oq=putnam+valley+covid+stats&aqs=chrome..69i57.7335j0j7&sourceid=chrome&ie=UTF-8)
2. [Putnam County Covid Website](https://www.putnamcountyny.com/covid19/)
3. [Weather.com Analytics](https://weather.com/coronavirus/l/f0659461d277de3751b2df36a487199ab73103b4feeb752dc157a8388b5db2ba)
4. And more, Having trouble finding vaccine numbers

### Scrapy Examlpe Program:
``` python
import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'http://quotes.toscrape.com/page/1/',
            'http://quotes.toscrape.com/page/2/',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = f'quotes-{page}.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log(f'Saved file {filename}')
```

## LSTM RNN Neural Network:
### Overview:
- Using Tensor Flow in Python
- Take data in Data.JSON returned from Scraper.py
- Split Data into Training and Testing 
- Train the Model then test and return its accuracy
- Optomize the Model based on the accuracy 
- Run a prediction on all of the Data
- Return Preditcion and accuracy into Predict.JSON file
- Use LSTM for model
### References:
1. [Tensor Flow CNN Docs](https://www.tensorflow.org/tutorials/images/cnn)
2. [Tensor Flow Docs](https://www.tensorflow.org/api_docs)
3. [Stock Predictor with Tensor Flow](https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877)
4. [2nd Stock predictor Refernce with Tensor Flow](https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras)
5. [Zika Virus CNN example](https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-019-1389-3)
6. [COVID Prediction Model](https://aws.amazon.com/blogs/machine-learning/introducing-the-covid-19-simulator-and-machine-learning-toolkit-for-predicting-covid-19-spread/)

### Example Code for CNN:
``` python
# Import data
data = pd.read_csv('data_stocks.csv')
# Drop date variable
data = data.drop(['DATE'], 1)
# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]
# Make data a numpy array
data = data.values

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# Scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)
# Build X and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# Import TensorFlow
import tensorflow as tf

# Define a and b as placeholders
a = tf.placeholder(dtype=tf.int8)
b = tf.placeholder(dtype=tf.int8)

# Define the addition
c = tf.add(a, b)

# Initialize the graph
graph = tf.Session()

# Run the graph
graph.run(c, feed_dict={a: 5, b: 4})

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Model architecture parameters
n_stocks = 500
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1
# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Make Session
net = tf.Session()
# Run initializer
net.run(tf.global_variables_initializer())

# Setup interactive plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.5)
plt.show()

# Number of epochs and batch size
epochs = 10
batch_size = 256

for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 5) == 0:
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
            plt.savefig(file_name)
            plt.pause(0.01)
# Print final MSE after Training
mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
print(mse_final)
```

## Website Display:
### Overview:
- Use Django For the website framework to make it easier with python applications
- Use bootstrap, HTML and CSS to create a good looking website to present
- Setup an App in Django and create a UI based on the Models output Data
- Create a SQL-lite database to store current and past model predictions and accuracy so the user and admins can pull up prediction history to see real accuracy of model
- Generate data and graphs based on COVID predictions and real scraped data based on the Data.JSON and Predict.JSON files
- If wanted: Host the website
### Examples:
- Some examples of Django websites include ../../django tutorial and ../../Personal Website (Must be in python Dir to view these files)
### References
1. [Django Docs](https://docs.djangoproject.com/en/3.1/)
2. [Django Example](https://realpython.com/get-started-with-django-1/)
3. [Django Youtube Tutorial](https://www.youtube.com/playlist?list=PLzMcBGfZo4-kQkZp-j9PNyKq7Yw5VYjq9)
4. [SQL-Lite Docs](https://sqlite.org/docs.html)

## Implementation Plan:
### Scraper:
#### Who:
- Alex and TBD
#### Plan:
- TBD
### Prediction Model:
#### Who:
- Aidan and Rowen 
#### Plan:
- Create Minimal Possible working RNN(LSTM) with Tensor Flow and get accuracy
- Upgrade Model parameters and layers to be optomized for COVID-19 spread prediction 
- Save model and allow to optomize itself each time its trained
- Output prediction to Predict.JSON file and store in Website Database each time a new prediction is made
### Website Display:
#### Who:
- TBD and TBD
#### Plan:
- TBD