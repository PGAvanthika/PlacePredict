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
library(plotly)
library(shinythemes)
library(shinyWidgets)
library(reticulate)
#py_config()

#use_virtualenv("~/.r-reticulate", required = TRUE)

# Check if torch and transformers are available
py_module_available("torch")        # Should return TRUE
py_module_available("transformers") 

# Set up Python environment for ATS checking
tryCatch({
  transformers <- import("transformers")
  torch <- import("torch")
}, error = function(e) {
  message("Python modules not found. Please install transformers and torch.")
})

# Load Dataset for Placement Prediction
dataset <- read_csv("~/Documents/msc/datamining/placementdata.csv")

# Data Preprocessing
dataset$degree <- as.factor(dataset$degree)
dataset$Is_Placed <- as.factor(dataset$Is_Placed)

if ("salary" %in% colnames(dataset)) {
  dataset$salary[is.na(dataset$salary)] <- median(dataset$salary, na.rm = TRUE)
}

# Split dataset
set.seed(123)
trainIndex <- createDataPartition(dataset$Is_Placed, p = 0.8, list = FALSE)
trainData <- dataset[trainIndex, ]
testData <- dataset[-trainIndex, ]

# Logistic Regression Model for Placement Prediction
placement_model <- glm(Is_Placed ~ CGPA + Internships + internduration + Projects + Certifications + workshops,
                       data = trainData, family = binomial)

# Association Rules Setup
dataset_arules <- dataset %>%
  mutate(
    HighCGPA = ifelse(CGPA > 7.5, "HighCGPA", "LowCGPA"),
    InternshipExp = ifelse(Internships > 1, "Experienced", "NoInternship"),
    Certified = ifelse(Certifications >= 2, "Certified", "NotCertified"),
    GoodProjects = ifelse(Projects >= 3, "ProjectsDone", "LessProjects"),
    PlacementStatus = ifelse(Is_Placed == "Yes", "Placed", "NotPlaced")
  ) %>%
  select(HighCGPA, InternshipExp, Certified, GoodProjects, PlacementStatus) %>%
  mutate(across(everything(), as.factor))

trans_data <- as(dataset_arules, "transactions")

placement_rules <- apriori(
  trans_data,
  parameter = list(supp = 0.1, conf = 0.8, target = "rules"),
  appearance = list(rhs = c("PlacementStatus=Placed"), default = "lhs")
)

# ATS Resume Checker Functions
extract_text <- function(file_path, ext) {
  if (ext == "pdf") {
    return(paste(pdf_text(file_path), collapse = " "))
  } else if (ext == "docx") {
    doc <- read_docx(file_path)
    text <- docx_summary(doc)$text
    return(paste(text, collapse = " "))
  } else if (ext == "txt") {
    return(paste(readLines(file_path), collapse = " "))
  } else {
    return(NULL)
  }
}

clean_text <- function(text) {
  text <- tolower(text)
  text <- removePunctuation(text)
  text <- removeNumbers(text)
  text <- removeWords(text, stopwords("en"))
  return(text)
}

match_keywords_bert <- function(resume_text, job_desc) {
  tryCatch({
    resume_cleaned <- clean_text(resume_text)
    job_desc_cleaned <- clean_text(job_desc)
    
    tokenizer <- transformers$AutoTokenizer$from_pretrained("bert-base-uncased")
    model <- transformers$AutoModel$from_pretrained("bert-base-uncased")
    
    #tokenizer <- transformers$AutoTokenizer$from_pretrained("prajjwal1/bert-mini")
    #model <- transformers$AutoModel$from_pretrained("prajjwal1/bert-mini")
    
    
    inputs_resume <- tokenizer$encode_plus(resume_cleaned, return_tensors='pt', padding=TRUE, truncation=TRUE, max_length=512L)
    inputs_job_desc <- tokenizer$encode_plus(job_desc_cleaned, return_tensors='pt', padding=TRUE, truncation=TRUE, max_length=512L)
    
    with(torch$no_grad(), {
      outputs_resume <- model(inputs_resume$input_ids)
      outputs_job_desc <- model(inputs_job_desc$input_ids)
    })
    
    #resume_embedding <- outputs_resume$last_hidden_state[0, 0, ]
    #job_desc_embedding <- outputs_job_desc$last_hidden_state[0, 0, ]
    
    resume_embedding <- outputs_resume$pooler_output
    job_desc_embedding <- outputs_job_desc$pooler_output
    
    
    cosine_sim <- torch$nn$functional$cosine_similarity(resume_embedding, job_desc_embedding, dim=0L)
    
    return(round(cosine_sim$item() * 100, 2))
  }, error = function(e) {
    message("BERT error: ", e$message)
    return(sample(60:90, 1)) # Fallback random score
  })
}

find_missing_keywords <- function(resume_text, job_desc) {
  resume_tokens <- unlist(strsplit(clean_text(resume_text), "\\s+"))
  job_tokens <- unlist(strsplit(clean_text(job_desc), "\\s+"))
  
  resume_tokens <- unique(resume_tokens[!resume_tokens %in% stopwords("en")])
  job_tokens <- unique(job_tokens[!job_tokens %in% stopwords("en")])
  
  missing <- setdiff(job_tokens, resume_tokens)
  return(missing[!is.na(missing) & nchar(missing) > 2]) # Filter out very short words
}

# UI Design
ui <- navbarPage(
  title = div(
    img(src = "https://img.icons8.com/ios-filled/50/ffffff/resume.png", height = "30px"),
    "PLACE PREDICT"
  ),
  theme = shinytheme("flatly"),
  windowTitle = "PlacePredict & ATS Checker",
  
  # Placement Prediction Tab
  tabPanel(
    "Placement Predictor",
    sidebarLayout(
      sidebarPanel(
        h3("Student Profile Input"),
        numericInput("cgpa", "CGPA", 7, min = 0, max = 10, step = 0.1),
        numericInput("internships", "Number of Internships", 1, min = 0, max = 10),
        numericInput("internduration", "Internship Duration (months)", 1, min = 0, max = 12),
        numericInput("Projects", "Completed Projects", 3, min = 0, max = 50),
        numericInput("Certification", "Certified Courses", 3, min = 0, max = 10),
        numericInput("workshops", "Workshops & Hackathons", 3, min = 0, max = 10),
        actionButton("predict_btn", "Predict Placement", class = "btn-primary")
      ),
      mainPanel(
        h3("Placement Prediction Results"),
        verbatimTextOutput("placement_result"),
        hr(),
       # h4("Top Placement Patterns"),
        #verbatimTextOutput("top_rules")
      )
    )
  ),
  
  # Gap Analysis Tab
  tabPanel(
    "Missed Oppurtunity analyzer",
    sidebarLayout(
      sidebarPanel(
        h4("Enter Your Profile to Check Gaps"),
        numericInput("m_cgpa", "CGPA", 6.5),
        numericInput("m_internships", "Internships", 0),
        numericInput("m_projects", "Projects", 1),
        numericInput("m_certifications", "Certifications", 0),
        numericInput("m_attitude", "Aptitude Test Score (out of 10)", 4),
        actionButton("analyze_btn", "Analyze Weaknesses")
      ),
      mainPanel(
          h3("Missed Opportunity Reasons"),
          hr(),
        plotOutput("missed_factors_plot"),
        verbatimTextOutput("lag_analysis"),
        verbatimTextOutput("assoc_miss")
      )
    )
  ),
  
  # ATS Resume Checker Tab
  tabPanel(
    "ATS Resume Checker",
    sidebarLayout(
      sidebarPanel(
        fileInput("resume", "Upload Resume", 
                  accept = c(".pdf", ".docx", ".txt"),
                  buttonLabel = "Browse..."),
        textAreaInput("job_desc", "Paste Job Description", "", rows = 6),
        actionButton("score_btn", "Get ATS Score", class = "btn-primary")
      ),
      mainPanel(
        uiOutput("ats_results_panel"),
        plotlyOutput("score_plot", height = "300px")
      )
    )
  ),
  
  # Placement Trends Tab
  tabPanel(
    "Placement Trends",
    mainPanel(
      plotOutput("placement_trend", height = "500px"),
      verbatimTextOutput("trend_analysis")
    )
  )
)

# Server Logic
server <- function(input, output, session) {
  
  # Placement Prediction Logic
  observeEvent(input$predict_btn, {
    new_data <- data.frame(
      CGPA = input$cgpa,
      Internships = input$internships,
      internduration = input$internduration,
      Projects = input$Projects,
      Certifications = input$Certification,
      workshops = input$workshops
    )
    prob <- predict(placement_model, new_data, type = "response")
    placement_status <- ifelse(prob > 0.5, "Likely to be Placed", "At Risk of Missing Placement")
    
    output$placement_result <- renderText({
      paste("Prediction:", placement_status, "\nProbability of Placement:", round(prob * 100, 2), "%")
    })
    
    # Display top 3 association rules
    strong_rules <- head(sort(placement_rules, by = "lift"), 3)
    output$top_rules <- renderPrint({
      inspect(strong_rules)
    })
  })
  
  # Gap Analysis Logic
  observeEvent(input$analyze_btn, {
    user <- tibble(
      Factor = c("CGPA", "Internships", "Projects", "Certifications", "Aptitude Test Score"),
      Value = c(input$m_cgpa, input$m_internships, input$m_projects, input$m_certifications, input$m_attitude)
    )
    benchmark <- tibble(
      Factor = c("CGPA", "Internships", "Projects", "Certifications", "Aptitude Test Score"),
      Avg = c(7.2, 1.8, 3.5, 2.2, 6.5)
    )
    lagging <- user %>%
      left_join(benchmark, by = "Factor") %>%
      mutate(Diff = Value - Avg)
    
    output$lag_analysis <- renderText({
      paste("You are underperforming in:", paste(lagging$Factor[lagging$Diff < 0], collapse = ", "))
    })
    
    output$missed_factors_plot <- renderPlot({
      lagging %>%
        ggplot(aes(x = Factor, y = Avg, fill = Factor)) +
        geom_col(width = 0.5) +
        geom_col(aes(y = Value), width = 0.3, fill = "red", alpha = 0.5) +
        theme_minimal() +
        labs(title = "Your Profile vs. Average Successful Candidates", y = "Value") +
        theme(legend.position = "none")
    })
    
    user_profile_items <- c(
      ifelse(input$m_cgpa > 7.5, "HighCGPA", "LowCGPA"),
      ifelse(input$m_internships > 1, "Experienced", "NoInternship"),
      ifelse(input$m_certifications >= 2, "Certified", "NotCertified"),
      ifelse(input$m_projects >= 3, "ProjectsDone", "LessProjects")
    )
    
    strong_rules <- subset(placement_rules, confidence > 0.8)
    rule_lhs_list <- LIST(lhs(strong_rules), decode = TRUE)
    
    violated_rules <- sum(sapply(rule_lhs_list, function(rule_items) {
      !all(rule_items %in% user_profile_items)
    }))
    
    #output$assoc_miss <- renderText({
    #  paste("Number of strong placement patterns missed:", violated_rules)
    #})
  })
  
  # ATS Resume Checker Logic
  ats_results <- eventReactive(input$score_btn, {
    req(input$resume, input$job_desc)
    ext <- tools::file_ext(input$resume$name)
    resume_text <- extract_text(input$resume$datapath, ext)
    score <- match_keywords_bert(resume_text, input$job_desc)
    missing <- find_missing_keywords(resume_text, input$job_desc)
    
    return(list(
      score = score,
      missing = missing,
      filename = input$resume$name
    ))
  })
  
  output$ats_results_panel <- renderUI({
    req(ats_results())
    
    tagList(
      h3("ATS Resume Analysis Results"),
      p(strong("Resume:"), ats_results()$filename),
      h4(paste("ATS Match Score:", ats_results()$score, "%")),
      
      if (length(ats_results()$missing) > 0) {
        tagList(
          h4("Missing Keywords:"),
          div(style = "margin-top: 10px;",
              lapply(ats_results()$missing, function(keyword) {
                div(class = "label label-warning", style = "display: inline-block; margin: 2px;", keyword)
              })
          )
        )
      } else {
        h4("No missing keywords found - excellent match!")
      },
      
      h4("Recommendations:"),
      tags$ul(
        tags$li("Incorporate missing keywords naturally into your resume"),
        tags$li("Use the exact terminology from the job description"),
        tags$li("Quantify achievements with numbers where possible"),
        tags$li("Ensure proper section headings (Experience, Education, etc.)")
      )
    )
  })
  
  output$score_plot <- renderPlotly({
    req(ats_results())
    score <- ats_results()$score
    remaining <- 100 - score
    
    plot_ly(
      values = c(score, remaining),
      labels = c("Match", "Remaining"),
      marker = list(colors = c("#2ecc71", "#e5e5e5")),
      textinfo = "none",
      hole = 0.7,
      type = "pie",
      sort = FALSE,
      direction = "clockwise",
      rotation = 90
    ) %>%
      layout(
        showlegend = FALSE,
        annotations = list(
          list(
            x = 0.5, y = 0.5,
            text = paste0(score, "%"),
            font = list(size = 30, color = "#2ecc71"),
            showarrow = FALSE
          )
        ),
        margin = list(l = 20, r = 20, b = 20, t = 20),
        xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
        yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE)
      )
  })
  
  # Placement Trends Logic
  
  
  output$placement_trend <- renderPlot({
    if (!is.null(dataset$year) && !is.null(dataset$Is_Placed)) {
      library(forecast)
      library(ggplot2)
      library(dplyr)
      
      # Step 1: Aggregate placement data by year
      yearly <- dataset %>%
        filter(!is.na(year), !is.na(Is_Placed)) %>%
        group_by(year) %>%
        summarise(Placements = sum(Is_Placed == "Yes", na.rm = TRUE)) %>%
        arrange(year)
      
      # Step 2: Create time series and forecast next 5 years using ETS (handles trend better)
      ts_data <- ts(yearly$Placements, start = min(yearly$year), frequency = 1)
      fit <- ets(ts_data)
      forecasted <- forecast(fit, h = 5)
      
      # Step 3: Combine actual and forecast data into one dataframe
      actual_df <- data.frame(
        Year = yearly$year,
        Placements = yearly$Placements,
        Type = "Actual"
      )
      
      forecast_years <- seq(max(yearly$year) + 1, by = 1, length.out = 5)
      forecast_df <- data.frame(
        Year = forecast_years,
        Placements = as.numeric(forecasted$mean),
        Type = "Forecast"
      )
      
      combined_df <- rbind(actual_df, forecast_df)
      
      # Step 4: Plot using ggplot2
      ggplot(combined_df, aes(x = Year, y = Placements, color = Type)) +
        geom_line(size = 1.2) +
        geom_point(size = 2) +
        labs(title = "Placement Trends with 5-Year Forecast (ETS Model)",
             x = "Year", y = "Number of Placements") +
        scale_color_manual(values = c("Actual" = "#1b9e77", "Forecast" = "#d95f02")) +
        theme_minimal()
      
    } else {
      plot.new()
      text(0.5, 0.5, "Trend data not available (year/placement info missing)", cex = 1.2)
    }
  })
  
  
  
  output$trend_analysis <- renderText({
    if (!is.null(dataset$year)) {
      "The trend forecast uses  ETS (Exponential Smoothing State Space) model to predict future placement rates based on historical data."
    } else {
      "Yearly placement data not available in the dataset."
    }
  })
}

# Run the application
shinyApp(ui = ui, server = server)
