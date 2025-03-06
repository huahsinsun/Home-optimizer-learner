function extracted_data = read_excel_files(folder_path, row_number, column_number)
    files = dir(fullfile(folder_path, '*.xlsx'));
    num_files = length(files);

    % Preallocate matrix to store extracted data
    extracted_data = zeros(0, 2);

    for i = 1:num_files
        % Read data from current file
        [~, ~, raw] = xlsread(fullfile(folder_path, files(i).name));

        % Extract columns O and P (without first column)
        extracted_data_i = cell2mat(raw(row_number, column_number));

        % Append extracted data to matrix
        extracted_data = [extracted_data extracted_data_i];
    end
end