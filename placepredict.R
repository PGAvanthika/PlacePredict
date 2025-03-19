library(shiny)
library(tidyverse)
library(caret)
library(forecast)
library(httr)
library(jsonlite)
library(readr)
library(tm)
library(pdftools)
library(arules)

# Load Dataset
dataset <- read_csv("/Users/avanthika/Desktop/placementdata.csv")

# Convert categorical variables to factors
dataset$degree <- as.factor(dataset$degree)
dataset$Is_Placed <- as.factor(dataset$Is_Placed)

# Handle missing salary values
if("Salary" %in% colnames(dataset)) {
  dataset$Salary[is.na(dataset$Salary)] <- median(dataset$Salary, na.rm = TRUE)
}

# Split data for model training
set.seed(123)
trainIndex <- createDataPartition(dataset$Is_Placed, p = 0.8, list = FALSE)
trainData <- dataset[trainIndex,]
testData <- dataset[-trainIndex,]

# Train logistic regression model for placement prediction
placement_model <- glm(Is_Placed ~ ., data = trainData, family = binomial)

# Forecasting job market trends
if("Year" %in% colnames(dataset)) {
  placement_trend <- dataset %>%
    group_by(Year) %>%
    summarize(Placed_Students = sum(as.numeric(Is_Placed)))
  
  placement_ts <- ts(placement_trend$Placed_Students, start = min(dataset$Year), frequency = 1)
  arima_model <- auto.arima(placement_ts)
  forecast_values <- forecast(arima_model, h = 5)
}

# Function to calculate ATS resume score using an API
get_ats_score <- function(resume_text, job_role) {
  url <- "https://api.example.com/ats-score"
  response <- POST(url, body = list(text = resume_text, job_role = job_role), encode = "json")
  result <- fromJSON(content(response, as = "text"))
  return(result$score)
}

# Function to extract text from resume PDF
extract_resume_text <- function(pdf_path) {
  pdf_text(pdf_path) %>%
    paste(collapse = " ") %>%
    tolower()
}

# UI with Custom Styling
ui <- fluidPage(
  tags$head(
    tags$style(HTML(
      "body { background-color: #c9d3b9; font-family: Arial, sans-serif; }
      .navbar { background-color: #98ab7f !important; }
      .tab-content { background-color: white; padding: 15px; border-radius: 10px; }
      .btn-primary { background-color: #98ab7f; border-color: #98ab7f; }
      .btn-primary:hover { background-color: #7d906a; }"
    ))
  ),
  titlePanel("Placement Prediction & Readiness System"),
  navbarPage("Menu",
             tabPanel("Placement Prediction",
                      sidebarLayout(
                        sidebarPanel(
                          numericInput("cgpa", "Enter CGPA", 7, min = 0, max = 10, step = 0.1),
                          numericInput("internships", "Number of Internships", 1, min = 0, max = 10),
                          actionButton("predict_btn", "Predict Placement")
                        ),
                        mainPanel(
                          textOutput("placement_result"),
                          plotOutput("placement_plot")
                        )
                      )
             ),
             tabPanel("Salary Prediction",
                      mainPanel(
                        textOutput("salary_prediction"),
                        plotOutput("salary_plot")
                      )
             ),
             tabPanel("ATS Resume Scoring",
                      sidebarPanel(
                        fileInput("resume", "Upload Resume (PDF)", accept = ".pdf"),
                        textInput("job_role", "Enter Job Role for ATS Scoring"),
                        actionButton("score_btn", "Get ATS Score")
                      ),
                      mainPanel(textOutput("ats_score"))
             ),
             tabPanel("Placement Trends",
                      mainPanel(plotOutput("placement_trend"))
             )
  )
)

# Server Logic
server <- function(input, output) {
  observeEvent(input$predict_btn, {
    new_data <- data.frame(CGPA = input$cgpa, Internships = input$internships)
    pred <- predict(placement_model, new_data, type = "response")
    placement_status <- ifelse(pred > 0.5, "Placed", "Not Placed")
    output$placement_result <- renderText(paste("Placement Prediction:", placement_status))
  })
  observeEvent(input$score_btn, {
    if (!is.null(input$resume$datapath)) {
      resume_text <- extract_resume_text(input$resume$datapath)
      score <- get_ats_score(resume_text, input$job_role)
      output$ats_score <- renderText(paste("ATS Resume Score:", score))
    }
  })
}

shinyApp(ui, server)
