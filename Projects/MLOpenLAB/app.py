import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import ollama
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Create the main window
window = tk.Tk()
window.geometry('1000x800')
window.title("Data Processing, Modeling, and Interaction")

# Global variables to store the dataframe and its features
df = pd.DataFrame()
df_scaled = pd.DataFrame()
df_model = None
dmodel = None
x_train, x_test, y_train, y_test = None, None, None, None
pred = None
numerical_features = []
categorical_features = []
user_name = ""

# Function to load a CSV file
def load_csv():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    
    if file_path:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        print("CSV file loaded successfully!")
        
        # Display the first few rows in the GUI
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, df.head().to_string())
        
        # Update the target variable dropdown with column names
        target_var.set('')  # Clear the current selection
        target_dropdown['menu'].delete(0, 'end')
        for col in df.columns:
            target_dropdown['menu'].add_command(label=col, command=tk._setit(target_var, col))
        
        # Update the x and y dropdowns for visualization
        x_var.set('')
        y_var.set('')
        x_dropdown['menu'].delete(0, 'end')
        y_dropdown['menu'].delete(0, 'end')
        for col in df.columns:
            x_dropdown['menu'].add_command(label=col, command=tk._setit(x_var, col))
            y_dropdown['menu'].add_command(label=col, command=tk._setit(y_var, col))

        # Display the notification pop-up
        messagebox.showinfo("About the Program", 
                            "The program is created by Faiz Raza, a BS Electrical Engineering student at the University of the Punjab.\n\n"
                            "Faiz is looking for collaborators to improve the program and invites you to connect for collaboration.")

# Function to fill null values and display the updated dataframe
def fill_null_values():
    global df
    global numerical_features
    global categorical_features
    
    text_widget.delete(1.0, tk.END)  # Clear the text widget
    
    numerical_features = [cols for cols in df.columns if df[cols].dtype in ['int64', 'float64']]
    categorical_features = [cols for cols in df.columns if df[cols].dtype == 'object']
    
    # Fill numerical features with mean
    for col in numerical_features:
        df[col] = df[col].fillna(df[col].mean())
    
    # Fill categorical features with mode
    for col in categorical_features:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    text_widget.insert(tk.END, "Null values have been filled.\n")
    text_widget.insert(tk.END, "Numerical Features filled with mean:\n")
    text_widget.insert(tk.END, ", ".join(numerical_features) + "\n\n")
    text_widget.insert(tk.END, "Categorical Features filled with mode:\n")
    text_widget.insert(tk.END, ", ".join(categorical_features) + "\n\n")
    text_widget.insert(tk.END, "Updated DataFrame (first 5 rows):\n")
    text_widget.insert(tk.END, df.head().to_string())

# Function to encode categorical features
def encoding():
    global df
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END, '### Encoding the Categorical Features\n')
    
    for i in categorical_features:
        enco = LabelEncoder()
        df[i] = enco.fit_transform(df[i])
    
    text_widget.insert(tk.END, "Encoded Dataset (first 5 rows):\n")
    text_widget.insert(tk.END, df.head().to_string())

# Function to scale the dataframe
def scaling(preprocess):
    global df
    global df_scaled
    scaler = None
    
    text_widget.delete(1.0, tk.END)
    
    if preprocess == 'Standard Scaler':
        scaler = StandardScaler()
    elif preprocess == 'Min-Max Scaler':
        scaler = MinMaxScaler()
    elif preprocess == 'Robust Scaler':
        scaler = RobustScaler()
    
    if scaler:
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        text_widget.insert(tk.END, f"### Scaled Dataset using {preprocess}\n")
        text_widget.insert(tk.END, df_scaled.head().to_string())

# Visualization function
def visulize(x, y, plot):
    x_data = df[x]
    y_data = df[y] if y else None
    
    if plot == 'Line Plot':
        plt.figure(figsize=(12, 8))
        plt.plot(x_data, y_data)
        plt.title(f'{x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        plt.show()
    elif plot == 'Scatter Plot':
        plt.figure(figsize=(12, 8))
        plt.scatter(x_data, y_data)
        plt.title(f'{x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        plt.show()
    elif plot == 'Box Plot':
        plt.figure(figsize=(12, 8))
        plt.boxplot(x_data)
        plt.title(f'{x}')
        plt.xlabel(x)
        plt.tight_layout()
        plt.show()
    elif plot == 'Hist2D':
        plt.figure(figsize=(12, 8))
        plt.hist2d(x_data, y_data, bins=30)
        plt.title(f'{x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        plt.show()
    elif plot == 'Stem':
        plt.figure(figsize=(12, 8))
        plt.stem(x_data, y_data)
        plt.title(f'{x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        plt.show()
    elif plot == 'Hist':
        plt.figure(figsize=(12, 8))
        plt.hist(x_data, bins=30)
        plt.title(f'{x}')
        plt.xlabel(x)
        plt.tight_layout()
        plt.show()
    elif plot == 'Bar':
        plt.figure(figsize=(12, 8))
        plt.bar(x_data, y_data)
        plt.title(f'{x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        plt.show()

# Function to save the entered text to a variable and generate a response
def save_name(event=None):
    global user_name
    user_name = name_entry.get()
    print(f"Name saved: {user_name}")
    generate_response()

# Function to generate a response using ollama
def generate_response():
    operation = "say 'We know you have got some awesome ideas brewing - let us turn them into reality! Welcome aboard!' to "
    prompt = operation + user_name

    response = ollama.chat(model='moondream', messages=[
      {
        'role': 'user',
        'content': prompt,
      },
    ])

    # Displaying the response in the GUI
    response_text_widget = tk.Text(window, height=5, width=50)
    response_text_widget.pack(pady=10)
    response_text_widget.insert(tk.END, response['message']['content'])

# Function to select and initialize the model
def select_model(model, degree=None):
    global df_model
    if model == 'Linear Regression':
        df_model = LinearRegression()
    elif model == 'Logistic Regression':
        df_model = LogisticRegression(max_iter=1000)
    elif model == 'Ridge':
        df_model = Ridge()
    elif model == 'ElasticNet':
        df_model = ElasticNet()
    elif model == 'Lasso':
        df_model = Lasso()
    elif model == 'RandomForestClassifier':
        df_model = RandomForestClassifier()
    elif model == 'DecisionTreeClassifier':
        df_model = DecisionTreeClassifier()
    elif model == 'SVC':
        df_model = SVC()
    elif model == 'KNeighborsClassifier':
        df_model = KNeighborsClassifier()
    text_widget.insert(tk.END, f"\nSelected Model: {model}\n")

# Function to train the model
def model_training(target):
    global x, y, x_train, x_test, y_train, y_test, dmodel
    try:
        x = df_scaled.drop(target, axis=1)
        y = df_scaled[target]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
        if df_model is not None:
            dmodel = df_model.fit(x_train, y_train)
            text_widget.insert(tk.END, "Model training completed successfully.\n")
        else:
            text_widget.insert(tk.END, '**Choose the proper target with the proper model**\n')
            logging.error('df_model is None. Model fitting cannot be performed.')
    except Exception as e:
        logging.error(f'An error occurred during model training: {str(e)}')
        text_widget.insert(tk.END, f'An error occurred: {str(e)}\n')

# Function to make predictions
def prediction(inputs):
    global pred
    try:
        pred = dmodel.predict(inputs)
        text_widget.insert(tk.END, "Prediction:\n")
        text_widget.insert(tk.END, str(pred) + "\n")
    except Exception as e:
        logging.error(f'An error occurred during prediction: {str(e)}')
        text_widget.insert(tk.END, f'An error occurred during prediction: {str(e)}\n')

# Function to evaluate the model
def evaluation(evaluation_metric):
    global pred
    try:
        if evaluation_metric == 'MSE':
            eval_result = mean_squared_error(y_test, pred)
        elif evaluation_metric == 'MAE':
            eval_result = mean_absolute_error(y_test, pred)
        elif evaluation_metric == 'R2 Score':
            eval_result = r2_score(y_test, pred)
        
        text_widget.insert(tk.END, f"{evaluation_metric}: {eval_result}\n")
    except Exception as e:
        logging.error(f'An error occurred during evaluation: {str(e)}')
        text_widget.insert(tk.END, f'An error occurred during evaluation: {str(e)}\n')

# Create the name entry widgets
frame_name = tk.Frame(window)
frame_name.pack(pady=10)

name_text = tk.Label(master=frame_name, text='Let us know your name!')
name_text.pack()

name_entry = tk.Entry(master=frame_name, width=20)
name_entry.pack()

# Bind the Enter key to the save_name function
name_entry.bind("<Return>", save_name)

# Adjust the position of the left and right frames
# Add some padding to move them slightly above

# Create a frame for buttons on the left
left_frame = tk.Frame(window)
left_frame.pack(side=tk.LEFT, padx=10, pady=110, fill='y')  

# Create a frame for buttons on the right
right_frame = tk.Frame(window)
right_frame.pack(side=tk.RIGHT, padx=10, pady=45, fill='y') 

# Create a button to load the CSV file (Left)
load_button = tk.Button(left_frame, text="Load CSV", command=load_csv)
load_button.pack(pady=5)

# Create a button to fill null values (Left)
fill_button = tk.Button(left_frame, text="Fill Null Values", command=fill_null_values)
fill_button.pack(pady=5)

# Create a button to encode categorical features (Left)
encode_button = tk.Button(left_frame, text="Encode Categorical Features", command=encoding)
encode_button.pack(pady=5)

# Create a dropdown to select a scaling method (Left)
scaling_label = tk.Label(left_frame, text="Select Scaling Method:")
scaling_label.pack(pady=5)

scaling_options = ['Standard Scaler', 'Min-Max Scaler', 'Robust Scaler']
scaling_var = tk.StringVar(window)
scaling_var.set(scaling_options[0])

scaling_dropdown = ttk.OptionMenu(left_frame, scaling_var, *scaling_options)
scaling_dropdown.pack(pady=5)

# Create a button to scale the dataframe (Left)
scale_button = tk.Button(left_frame, text="Scale Data", command=lambda: scaling(scaling_var.get()))
scale_button.pack(pady=5)

# Create a dropdown to select a model (Right)
model_label = tk.Label(right_frame, text="Select Model:")
model_label.pack(pady=5)

model_options = ['Linear Regression', 'Logistic Regression', 'Ridge', 'ElasticNet', 'Lasso', 
                 'RandomForestClassifier', 'DecisionTreeClassifier', 'SVC', 'KNeighborsClassifier']
model_var = tk.StringVar(window)
model_var.set(model_options[0])

model_dropdown = ttk.OptionMenu(right_frame, model_var, *model_options)
model_dropdown.pack(pady=5)

# Create a button to select and initialize the model (Right)
select_model_button = tk.Button(right_frame, text="Select Model", command=lambda: select_model(model_var.get()))
select_model_button.pack(pady=5)

# Create a dropdown to select the target variable for model training (Right)
target_label = tk.Label(right_frame, text="Select Target Variable:")
target_label.pack(pady=5)

target_var = tk.StringVar(window)
target_var.set('')  # Set a default target variable or leave it empty

# This dropdown will be populated once the CSV is loaded
target_dropdown = ttk.OptionMenu(right_frame, target_var, '')
target_dropdown.pack(pady=5)

# Create a button to train the model (Right)
train_button = tk.Button(right_frame, text="Train Model", command=lambda: model_training(target_var.get()))
train_button.pack(pady=5)

# Create a button to make predictions (Right)
predict_button = tk.Button(right_frame, text="Make Predictions", command=lambda: prediction(x_test))
predict_button.pack(pady=5)

# Create a dropdown to select an evaluation metric (Right)
eval_label = tk.Label(right_frame, text="Select Evaluation Metric:")
eval_label.pack(pady=5)

eval_options = ['MSE', 'MAE', 'R2 Score']
eval_var = tk.StringVar(window)
eval_var.set(eval_options[0])

eval_dropdown = ttk.OptionMenu(right_frame, eval_var, *eval_options)
eval_dropdown.pack(pady=5)

# Create a button to evaluate the model (Right)
eval_button = tk.Button(right_frame, text="Evaluate Model", command=lambda: evaluation(eval_var.get()))
eval_button.pack(pady=5)

# Create a frame for the visualization options (Middle)
visualization_frame = tk.Frame(window)
visualization_frame.pack(pady=10)

# Create dropdowns for selecting X and Y columns for visualization
x_label = tk.Label(visualization_frame, text="Select X Variable:")
x_label.pack(side=tk.LEFT, padx=5)

x_var = tk.StringVar(window)
x_dropdown = ttk.OptionMenu(visualization_frame, x_var, '')
x_dropdown.pack(side=tk.LEFT, padx=5)

y_label = tk.Label(visualization_frame, text="Select Y Variable:")
y_label.pack(side=tk.LEFT, padx=5)

y_var = tk.StringVar(window)
y_dropdown = ttk.OptionMenu(visualization_frame, y_var, '')
y_dropdown.pack(side=tk.LEFT, padx=5)

# Create a dropdown to select the plot type
plot_label = tk.Label(visualization_frame, text="Select Plot Type:")
plot_label.pack(side=tk.LEFT, padx=5)

plot_options = ['Line Plot', 'Scatter Plot', 'Box Plot', 'Hist2D', 'Stem', 'Hist', 'Bar']
plot_var = tk.StringVar(window)
plot_var.set(plot_options[0])

plot_dropdown = ttk.OptionMenu(visualization_frame, plot_var, *plot_options)
plot_dropdown.pack(side=tk.LEFT, padx=5)

# Create a button to visualize the data
visualize_button = tk.Button(visualization_frame, text="Visualize", command=lambda: visulize(x_var.get(), y_var.get(), plot_var.get()))
visualize_button.pack(side=tk.LEFT, padx=5)

# Create a scrolled text widget for displaying the DataFrame and messages (Middle)
text_widget = tk.Text(window, wrap=tk.WORD, width=80, height=20)
text_widget.pack(pady=10)

# Add the collaboration message at the bottom of the window
collaboration_message = tk.Label(window, text="Looking for collaboration to make this better. faiz.raza.dec@gmail.com", 
                                 font=("Arial", 10), fg="blue")
collaboration_message.pack(side=tk.BOTTOM, pady=10)

# Run the main loop
window.mainloop()