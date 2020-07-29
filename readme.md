# Data+ 2020 Utility Forecasting for Duke University

Working with Duke’s Facilities Department, we have developed a tool which allows the cleaning and forecasting of utility usage data, for use in accurate budgeting and planning of new buildings.

## Usage
1. Clone this repo: `git clone https://github.com/epswartz/utility-forecasting.git`
2. Install packages: `pip install -r requirements.txt`
3. Run: `voila Forecasting_Tool.ipynb --VoilaConfiguration.file_whitelist="['.*\.csv']`
4. A browser window should open displaying the tool, but if not, you can navigate to http://localhost:8866 yourself.

### Sample Data
Sample data is provided in `sample_data/sample_data.csv`. To use this data, choose `15 Minute Data` for "Input Data Frequency" after you upload the data.
