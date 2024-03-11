# Ear_EEG_code

File Structure

--analysis
   
    --matlab
        --code_env
            --envelope_process_ear: the code of 3.1 "Stimulus reconstruction"
        --code_space
            --preprocess_IIR: preprocess data to prepare for ASAD
    
    --python
        --earcode
            --main.py: the code of 3.2 "ASAD"
            --config.py: change the 5 line, "model_name = model_names[3]# # you could change the code to other models by only changing the number"
    
    --statistic_analysis
        --results: the results achieved from python code.
        --plot_fig3.m: plot figure 3.

--preprocess
    --cuteeg.m: cut the precessed data from raw ear-EEG dara as described in 2.3

--preprocess_data
  * no file now: you could get the data after runing cuteeg.m

--rawdata
  * no file now: The raw data was released. Please download raw data from the following link:

--allcode.m: To help you run our code, you could run "allcode.m" to do everything before ASAD (ASAD is python code rather than matlab code).
