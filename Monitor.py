import streamlit as st
import pandas as pd
import numpy as np
import os
import calendar
import scipy.stats as scs
import spei as si
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
Scaler=StandardScaler()


# Add a sidebar for date selection
st.sidebar.header("Select Date Range")

# Default dates
default_start_date = datetime(2024, 1, 1)
default_end_date = datetime(2025, 12, 31)

# Create the calendar widgets for start and end dates
start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", default_end_date)

# Convert dates to strings in 'YYYY-MM-DD' format
startDate = start_date.strftime("%Y-%m-%d")
endDate = end_date.strftime("%Y-%m-%d")


# Ensure the end date is after the start date
if start_date > end_date:
    st.error("End date must be after start date.")
# Title of the Streamlit app
st.title('Sahel Cat Layer Project')

# Select the country from the dropdown
country = st.selectbox('Select the country you want to monitor', ('BurkinaFaso', 'Mali', 'Niger'))

# Select the crop from the dropdown
crop = st.selectbox('Select the crop you want to monitor', ('Millet', 'Sorghum', 'Maize'))

# Select the dataset from the dropdown
dataset = st.selectbox('Select the dataset to monitor', ('ERA5', 'CHIRPS'))

if dataset=='CHIRPS':

    # Upload the Excel file containing district names and weights
    uploaded_file = st.file_uploader("Upload the Excel file with district weights", type="xlsx")

    if uploaded_file is not None:
        # Load data from the uploaded Excel file
        country_crop_input = pd.read_excel(uploaded_file)
        # Create a dictionary with district names and their weights
        country_district_weightings = dict(zip(country_crop_input['Modified_Areas'], country_crop_input['Region_Weight']))

        # Paths for data storage
        data_path = f"C:\\Users\\PatrickMunyingi\\Data\\{dataset}\\District\\{country}\\{crop}\\"
        processed_data_path = f"C:\\Users\\PatrickMunyingi\\Data\\{dataset}\\District\\{country}\\{crop}\\Processed\\"
        results_path = f"C:\\Users\\PatrickMunyingi\\Data\\{dataset}\\District\\{country}\\{crop}\\Results\\"
        country_results_path = f"C:\\Users\\PatrickMunyingi\\Data\\{dataset}\\Country\\{country}\\{crop}\\Results\\Monthly_Output\\"

        # Function to process precipitation data
        def process_data(df_precip):
            """Prepare precipitation data by combining relevant columns."""
            if '.geo' in df_precip.columns:
                df_precip = df_precip.drop('.geo', axis=1)
            combined_df = pd.concat([df_precip['system:index'], df_precip.iloc[:, 1:]], axis=1)
            return combined_df

        # Function to prepare data for each district
        def data_prep(country_district_names):
            """Prepare data for each district, processing daily and monthly climate data."""
          
            for district_name in country_district_names:
                # Construct the file path based on the selected country and crop
                district_file = f'mean_{country}_{crop}_District_{district_name}_Precipitation_2024-01-01_to_2024-07-31.csv'
                district_file_path = os.path.join(data_path, district_file)

                if os.path.exists(district_file_path):
                    df_precip = pd.read_csv(district_file_path)
                    combined_df = process_data(df_precip)
                   

                    # Rename the 'system:index' column to 'Date' and parse it as datetime
                    combined_df = combined_df.rename(columns={'system:index': 'Date'})
                    combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%Y%m%d')
                    combined_df = combined_df.set_index('Date')

                    # Resample the data to monthly frequency
                    monthly_df = combined_df.resample('ME').mean()

                    # Save the processed data
                    combined_df.to_csv(processed_data_path + district_name + '_daily_climate_data.csv', index=True)
                    monthly_df.to_csv(processed_data_path + district_name + '_monthly_climate_data.csv', index=True)
                else:
                    st.error(f"File not found: {district_file_path}")

            return combined_df, monthly_df

        # Calculate the total number of months between two dates
        def calculate_total_months(start_date_str, end_date_str):
            """Calculate the total number of months between two dates."""
            # Convert string dates to datetime objects
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

            # Calculate the difference between the dates
            difference = relativedelta(end_date, start_date)

            # Calculate total months
            total_months = difference.years * 12 + difference.months

            return total_months

        # Function to calculate Standard Precipitation Index (SPI)
        def calculate_spi(file_path, threshold):
            """Calculate Standard Precipitation Index (SPI)."""
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # Ensure the index is in chronological order
            df.sort_index(inplace=True)

            # Calculate SPI
            df_prec = df['precipitation'].resample('ME').sum()  # Use .sum() for total precipitation per month
            rolling_sum = df_prec.rolling(window=threshold, min_periods=1).sum().dropna()
         
            spi = si.spi(rolling_sum, dist=scs.pearson3)
            spi_standardized = (spi - spi.mean()) / spi.std()

            print(f'The Mean of SPI is: {spi_standardized.mean()}')
            print(f"The Standard Deviation of SPI is: {spi_standardized.std()}")

            return spi_standardized

        # Export SPI data for each month
        def export_monthly_spi(month, output_folder, country, crop):
            """Export monthly SPI data to CSV."""
            month_name = calendar.month_name[month]
            for key in country_district_weightings.keys():
                month_spi = final_spi[[col for col in final_spi if col.endswith(f'_{key}')]].loc[final_spi.index.month == month]

                month_spi.index = month_spi.index.year.astype(str)
                month_spi = month_spi.T
                mean_month_spi=month_spi.mean()
                std_month_spi=month_spi.std()

                standardized_spi=(month_spi-mean_month_spi)/std_month_spi

                if not month_spi.empty:
                    output_file_spi = os.path.join(output_folder, f'{dataset}_{country}_{crop}_SPI_{key}_{month_name}.csv')
                    standardized_spi.to_csv(output_file_spi)

        # Export country-level SPI data for each month
        def export_country_spi(month, output_folder, country, crop):
            """Export country-level SPI data for each month."""
            month_name = calendar.month_name[month]
            country_month_spi = country_df.loc[country_df.index.month == month]

            country_month_spi.index = country_month_spi.index.year.astype(str)
            country_month_spi = country_month_spi.T

            if not country_month_spi.empty:
                output_file_country_spi = os.path.join(output_folder, f'{dataset}_{country}_{crop}_Country_SPI_{month_name}.csv')
                country_month_spi.to_csv(output_file_country_spi)

        # Add a button for data processing
        if st.button('Process the Data'):
            with st.spinner('Processing data... Please wait...'):
                # Prepare the data for the selected country and crop
                combined_df, monthly_df = data_prep(country_district_weightings.keys())

                # Calculate total months for the given date range
                total_months = calculate_total_months(startDate, endDate)
                final_spi = pd.DataFrame()

                # Process each district
                for key, value in country_district_weightings.items():
                    file_path = os.path.join(processed_data_path, f'{key}_daily_climate_data.csv')

                    if not os.path.exists(file_path):
                        st.error(f"Missing data file for district: {key}")
                        continue

                    times = [1,2,3]  # SPI calculation thresholds
                    for threshold in times:
                        spi = calculate_spi(file_path, threshold)
                        final_spi[f'SPI_{threshold}_{key}'] = spi

                # Export SPI for each month
                for month in range(5,7):  # June to November
                    export_monthly_spi(month, results_path, country, crop)
                

                # Save overall SPI data
                final_spi.to_csv(results_path + f'{dataset}_{country}_{crop}_monthly_non_weighted_final_spi.csv')

                # Country aggregation with weighting
                for key, value in country_district_weightings.items():
                    if f'SPI_1_{key}' in final_spi.columns:
                        final_spi[f'SPI_1_{key}'] *= value
                        #Standardization
                        final_spi[f'SPI_1_{key}'] = (final_spi[f'SPI_1_{key}'] - final_spi[f'SPI_1_{key}'].mean()) / final_spi[f'SPI_1_{key}'].std()

                    if f'SPI_2_{key}' in final_spi.columns:
                        final_spi[f'SPI_2_{key}'] *= value
                        #Standardization
                        final_spi[f'SPI_2_{key}'] = (final_spi[f'SPI_2_{key}'] - final_spi[f'SPI_2_{key}'].mean()) / final_spi[f'SPI_2_{key}'].std()

               
                # Save country data for SPI 6
                final_spi.to_csv(country_results_path + f'{dataset}_{country}_{crop}_SPI_Country_monthly_weighted_final_spi.csv')
                # Seperate SPI_6 and SPI_9 columns
                spi_1_columns = final_spi.filter(like='SPI_1')
                spi_2_columns = final_spi.filter(like='SPI_2')
                spi_3_columns = final_spi.filter(like='SPI_3')

                # Calculate sum and mean
                spi_1_columns.loc[:, 'Sum'] = spi_1_columns.sum(axis=1)
                spi_1_columns.loc[:, 'Mean'] = spi_1_columns.mean(axis=1)

                spi_2_columns.loc[:, 'Sum'] = spi_2_columns.sum(axis=1)
                spi_2_columns.loc[:, 'Mean'] = spi_2_columns.mean(axis=1)

                spi_3_columns.loc[:, 'Sum'] = spi_3_columns.sum(axis=1)
                spi_3_columns.loc[:, 'Mean'] = spi_3_columns.mean(axis=1)

                            # Create a new dataframe
                country_df = pd.DataFrame()

                    # Calculate sum and mean

                    # new_df['SPI_Sum'] = final_spi.sum(axis=1)
                country_df['SPI_1_Mean'] = spi_1_columns['Mean']
                country_df['SPI_2_Mean'] = spi_2_columns['Mean']
                country_df['SPI_3_Mean'] = spi_2_columns['Mean']


                country_df['SPI_1_Mean']=((spi_1_columns['Mean'])-np.mean(spi_1_columns['Mean']))/spi_1_columns['Mean'].std()
                country_df['SPI_2_Mean']=((spi_2_columns['Mean'])-np.mean(spi_2_columns['Mean']))/spi_2_columns['Mean'].std()
                country_df['SPI_3_Mean']=((spi_3_columns['Mean'])-np.mean(spi_3_columns['Mean']))/spi_3_columns['Mean'].std()
                

                country_df.to_csv(results_path + 'Chirps_monthly_country_spi.csv')


                # Export SPI data for each month
                for month in range(5, 7):
                    export_country_spi(month, country_results_path, country, crop)
            def standardize_rows(df):
                df_numeric = df.apply(pd.to_numeric, errors='coerce')
                def standardize_row(row):
                    mean = row.mean()
                    std = row.std()
                    if std == 0:
                        return np.zeros_like(row)
                    return (row - mean) / std

                standardized_df = df_numeric.apply(standardize_row, axis=1)
                standardized_df.index = df.index
                standardized_df.columns = df.columns
                return standardized_df

            

            for filename in os.listdir(country_results_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(country_results_path, filename)
                    df = pd.read_csv(file_path,index_col=0)
                    

                    # Apply the standardization function
                    standardized_df = standardize_rows(df)
                    
                    # Overwrite the original file with the standardized dataframe
                    standardized_df.to_csv(file_path, index=True)
                    

            # Display completion message
            st.balloons()
            st.success('Data processing is complete!')
            

else:
    # Upload the Excel file containing district names and weights
    uploaded_file = st.file_uploader("Upload the Excel file with district weights", type="xlsx")

    if uploaded_file is not None:
        # Load data from the uploaded Excel file
        country_crop_input = pd.read_excel(uploaded_file)
        # Create a dictionary with district names and their weights
        country_district_weightings = dict(zip(country_crop_input['Modified_Areas'], country_crop_input['Region_Weight']))

        # Paths for data storage
        data_path = f"C:\\Users\\PatrickMunyingi\\Data\\ERA5\\District\\{country}\\{crop}\\"
        processed_data_path = f"C:\\Users\\PatrickMunyingi\\Data\\ERA5\\District\\{country}\\{crop}\\Processed\\"
        results_path = f"C:\\Users\\PatrickMunyingi\\Data\\ERA5\\District\\{country}\\{crop}\\Results\\"
        country_results_path = f"C:\\Users\\PatrickMunyingi\\Data\\ERA5\\Country\\{country}\\{crop}\\Results\\Monthly_Output\\"


        def process_data(df_max_precip, df_min_precip, df_max_pe, df_min_pe):
            """Combine and process data from multiple sources."""
            
            # Drop or rename the .geo columns if they exist in your dataframes
            if '.geo' in df_max_precip.columns:
                df_max_precip = df_max_precip.drop(columns=['.geo'])
            if '.geo' in df_min_precip.columns:
                df_min_precip = df_min_precip.drop(columns=['.geo'])
            if '.geo' in df_max_pe.columns:
                df_max_pe = df_max_pe.drop(columns=['.geo'])
            if '.geo' in df_min_pe.columns:
                df_min_pe = df_min_pe.drop(columns=['.geo'])

            
            # Merge dataframes on the common column 'system:index'
            combined_df = pd.concat([df_max_precip['system:index'], 
                                df_max_pe.iloc[:, 1:], 
                                df_min_pe.iloc[:, 1:],  
                                df_min_precip.iloc[:, 1:], 
                                df_max_precip.iloc[:, 1:]], axis=1) 

            return combined_df

        def data_prep(country_district_names):
            """Prepare data for each district, processing daily and monthly climate data."""
           

            for district_name in country_district_names:
                if dataset == 'ERA5':
                    # Paths for ERA5 dataset
                    df_max_precip_path = os.path.join(
                        data_path,
                        f'mean_{country}_{crop}_District_{district_name}_maximumPrecipitation_{startDate}_to_{endDate}.csv'
                    )
                    df_min_precip_path = os.path.join(
                        data_path,
                        f'mean_{country}_{crop}_District_{district_name}_minimumPrecipitation_{startDate}_to_{endDate}.csv'
                    )
                    df_max_pe_path = os.path.join(
                        data_path,
                        f'mean_{country}_{crop}_District_{district_name}_maximumPotentialEvapotranspiration_{startDate}_to_{endDate}.csv'
                    )
                    df_min_pe_path = os.path.join(
                        data_path,
                        f'mean_{country}_{crop}_District_{district_name}_minimumPotentialEvapotranspiration_{startDate}_to_{endDate}.csv'
                    )

                    # Read the ERA5 data
                    df_max_precip = pd.read_csv(df_max_precip_path)
                    df_min_precip = pd.read_csv(df_min_precip_path)
                    df_max_pe = pd.read_csv(df_max_pe_path)
                    df_min_pe = pd.read_csv(df_min_pe_path)

                    # Combine data using the process_data function
                    combined_df = process_data(df_max_precip, df_min_precip, df_max_pe, df_min_pe)

                # Process the combined dataframe
               
                # Rename 'system:index' to 'Date' and parse it as datetime
                combined_df = combined_df.rename(columns={'system:index': 'Date'})
                combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%Y%m%d')
                combined_df = combined_df.set_index('Date')

                # Resample the data to monthly frequency
                monthly_df = combined_df.resample('ME').mean()

                # Save the processed data
                combined_df.to_csv(processed_data_path + district_name + '_daily_climate_data.csv', index=True)
                monthly_df.to_csv(processed_data_path + district_name + '_monthly_climate_data.csv', index=True)

            
            return combined_df, monthly_df

        
        def calculate_total_months(start_date_str, end_date_str):
            """Calculate the total number of months between two dates."""
            # Convert string dates to datetime objects
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

            # Calculate the difference between the dates
            difference = relativedelta(end_date, start_date)

            # Calculate total months
            total_months = difference.years * 12 + difference.months

            return total_months
        
        def caculate_spi_spei_combined(file_path, threshold):
            """Calculate combined SPI and SPEI."""
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Ensure the index is in chronological order
            df.sort_index(inplace=True)
            
           
            
            prec_min = df['total_precipitation_min'] 
            prec_max = df['total_precipitation_max']
            prec_mean = (prec_min + prec_max) / 2
            
            pe_min = df['potential_evaporation_min']
            pe_max = df['potential_evaporation_max']
            pe_mean = (pe_min + pe_max) / 2
            
            df_prec = (prec_mean.resample('ME').sum()) * 1000 # Monthly total precipitation # Convert to mm
            rolling_sum = df_prec.rolling(window=threshold, min_periods=1).sum().dropna()
            
            # Calculate SPI
            spi = si.spi(rolling_sum, dist=scs.gamma)  
                
            water_balance = (df_prec - pe_mean).dropna()
            spei = si.spei(water_balance.rolling(window=threshold, 
                                                min_periods=1).sum().dropna(), dist=scs.fisk)  
            
           

                        
            return spi, spei   
        def export_monthly_spi_spei(month, output_folder, country, crop):
            """Export monthly SPI and SPEI to CSV."""
            month_name = calendar.month_name[month]
            for key in country_district_weightings.keys():
                month_spi = final_spi[[col for col in final_spi if col.endswith(f'_{key}')]].loc[final_spi.index.month == month]
                month_spei = final_spei[[col for col in final_spei if col.endswith(f'_{key}')]].loc[final_spei.index.month == month]
                
                month_spi.index = month_spi.index.year.astype(str)
                month_spi = month_spi.T
                
                
                
                month_spei.index = month_spei.index.year.astype(str)
                month_spei = month_spei.T

    


                # print(month_spi)

                if not month_spi.empty:
                    output_file_spi = os.path.join(output_folder, f'{dataset}_{country}_{crop}_SPI_{key}_{month_name}.csv')
                    month_spi.to_csv(output_file_spi)
                    
                if not month_spei.empty:
                    output_file_spei = os.path.join(output_folder, f'{dataset}_{country}_{crop}_SPEI_{key}_{month_name}.csv')
                    month_spei.to_csv(output_file_spei)

        def export_country_spi_spei(month, output_folder, country, crop):
            """Export country-level SPI and SPEI data for each month."""
            month_name = calendar.month_name[month]
            country_month_spi = country_df.loc[country_df.index.month == month]
            country_month_spei = country_df.loc[country_df.index.month == month]


            country_month_spi.index = country_month_spi.index.year.astype(str)
            country_month_spei.index = country_month_spei.index.year.astype(str)
            country_month_spi = country_month_spi.T
            country_month_spei = country_month_spei.T
            
           

            if not country_month_spi.empty:
                output_file_country_spi = os.path.join(output_folder, f'{dataset}_{country}_{crop}_Country_SPI_{month_name}.csv')
                country_month_spi.to_csv(output_file_country_spi)

            if not country_month_spei.empty:
                output_file_country_spei = os.path.join(output_folder, f'{dataset}_{country}_{crop}_Country_SPEI_{month_name}.csv')
                country_month_spei.to_csv(output_file_country_spei)
        # Add a button for data processing
        if st.button('Process the Data'):
            with st.spinner('Processing data... Please wait...'):
                # Prepare the data for the selected country and crop
                combined_df, monthly_df = data_prep(country_district_weightings.keys())

                # Calculate total months for the given date range
                total_months = calculate_total_months(startDate, endDate)
                final_spi = pd.DataFrame()
                final_spei=pd.DataFrame()

                # Process each district
                for key, value in country_district_weightings.items():
                    file_path = os.path.join(processed_data_path, f'{key}_daily_climate_data.csv')

                    if not os.path.exists(file_path):
                        st.error(f"Missing data file for district: {key}")
                        continue

                    times = [6, 9]  # SPI calculation thresholds
                    for threshold in times:
                        spi,spei =caculate_spi_spei_combined(file_path, threshold)
                        final_spi[f'SPI_{threshold}_{key}'] = spi
                        final_spei[f'SPEI_{threshold}_{key}'] = spei
                
        # Export SPI and SPEI for each month
                for month in range(6, 12):  # June to November
                    export_monthly_spi_spei(month, results_path, country, crop)

                # Save overall SPI data
                final_spi.to_csv(results_path + f'{dataset}_{country}_{crop}_monthly_non_weighted_final_spi.csv')
                final_spei.to_csv(results_path + f'{dataset}_{country}_{crop}_monthly_non_weighted_final_spei.csv')
                # Country aggregation with weighting
                for key, value in country_district_weightings.items():
                    if f'SPI_6_{key}' in final_spi.columns:
                        final_spi[f'SPI_6_{key}'] *= value
                        final_spi[f'SPI_6_{key}'] = (final_spi[f'SPI_6_{key}'] - final_spi[f'SPI_6_{key}'].mean()) / final_spi[f'SPI_6_{key}'].std()

                    if f'SPI_9_{key}' in final_spi.columns:
                        final_spi[f'SPI_9_{key}'] *= value
                        final_spi[f'SPI_9_{key}'] = (final_spi[f'SPI_9_{key}'] - final_spi[f'SPI_9_{key}'].mean()) / final_spi[f'SPI_9_{key}'].std()

                    if f'SPEI_6_{key}' in final_spei.columns:
                        final_spei[f'SPEI_6_{key}'] *= value
                        final_spei[f'SPEI_6_{key}'] = (final_spei[f'SPEI_6_{key}'] - final_spei[f'SPEI_6_{key}'].mean()) / final_spei[f'SPEI_6_{key}'].std()

                    if f'SPEI_9_{key}' in final_spei.columns:
                        final_spei[f'SPEI_9_{key}'] *= value
                        final_spei[f'SPEI_9_{key}'] = (final_spei[f'SPEI_9_{key}'] - final_spei[f'SPEI_9_{key}'].mean()) / final_spei[f'SPEI_9_{key}'].std()

                # Seperate SPI_6 and SPI_9 columns
                spi_6_columns = final_spi.filter(like='SPI_6')
                spi_9_columns = final_spi.filter(like='SPI_9')

                # Seperate SPEI_6 and SPEI_9 columns
                spei_6_columns = final_spei.filter(like='SPEI_6')
                spei_9_columns = final_spei.filter(like='SPEI_9')
                
                # Calculate sum and mean
                spi_6_columns.loc[:, 'SPI_6_Sum'] = spi_6_columns.sum(axis=1)
                spi_6_columns.loc[:, 'SPI_6_Mean'] = spi_6_columns.mean(axis=1)

                spi_9_columns.loc[:, 'SPI_9_Sum'] = spi_9_columns.sum(axis=1)
                spi_9_columns.loc[:, 'SPI_9_Mean'] = spi_9_columns.mean(axis=1)

                # Calculate sum and mean
                spei_6_columns.loc[:, 'SPEI_6_Sum'] = spei_6_columns.sum(axis=1)
                spei_6_columns.loc[:, 'SPEI_6_Mean'] = spei_6_columns.mean(axis=1)

                spei_9_columns.loc[:, 'SPEI_9_Sum'] = spei_9_columns.sum(axis=1)
                spei_9_columns.loc[:, 'SPEI_9_Mean'] = spei_9_columns.mean(axis=1)


                # Save country data for SPEI 
                country_df=pd.DataFrame()

                country_df['SPEI_6_Mean'] = spei_6_columns['SPEI_6_Mean']
                country_df['SPEI_9_Mean'] = spei_9_columns['SPEI_9_Mean']
                
                ##Standardizing the Country Indices.
                country_df['SPEI_6_Mean']=((spei_6_columns['SPEI_6_Mean'])-np.mean(spei_6_columns['SPEI_6_Mean']))/spei_6_columns['SPEI_6_Mean'].std()
                country_df['SPEI_9_Mean']=((spei_9_columns['SPEI_9_Mean'])-np.mean(spei_9_columns['SPEI_9_Mean']))/spei_9_columns['SPEI_9_Mean'].std()
                
                # new_df['SPI_Sum'] = final_spi.sum(axis=1)
                country_df['SPI_6_Mean'] = spi_6_columns['SPI_6_Mean']
                country_df['SPI_9_Mean'] = spi_9_columns['SPI_9_Mean']
                
                #Standardizing the SPI Indices
                country_df['SPI_6_Mean']=((spi_6_columns['SPI_6_Mean'])-np.mean(spi_6_columns['SPI_6_Mean']))/spi_6_columns['SPI_6_Mean'].std()
                country_df['SPI_9_Mean']=((spi_9_columns['SPI_9_Mean'])-np.mean(spi_9_columns['SPI_9_Mean']))/spi_9_columns['SPI_9_Mean'].std()

                country_df.to_csv(results_path + 'ERA5_monthly_country_spei_spi.csv')

                # Export SPI data for each month
                for month in range(6, 12):
                    export_country_spi_spei(month, country_results_path, country, crop)
            def standardize_rows(df):
                    df_numeric = df.apply(pd.to_numeric, errors='coerce')
                    def standardize_row(row):
                        mean = row.mean()
                        std = row.std()
                        if std == 0:
                            return np.zeros_like(row)
                        return (row - mean) / std

                    standardized_df = df_numeric.apply(standardize_row, axis=1)
                    standardized_df.index = df.index
                   
                    return standardized_df


            for filename in os.listdir(country_results_path):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(country_results_path, filename)
                        df = pd.read_csv(file_path,index_col=0)
                        
                        # Apply the standardization function
                        standardized_df = standardize_rows(df)
                        
                        # Overwrite the original file with the standardized dataframe
                        standardized_df.to_csv(file_path, index=True)

            
                # Display completion message
            st.balloons()
            st.success('Data processing is complete!')
           











