%Catalogues the IR stats for the generated IRs into a .txt file with name
%filename.
clear all
listing = dir;
n_items = length(listing);
filename = 'catalogue.txt';
% This is the formatting of the csv file
A = {'Source Folder','Index 1','Index 2','I/O', 'RT' ,'DRR' ,'CTE', 'CFS' ,'EDT'};
% Start from scratch every time the file is run.
writecell(A, filename,'FileType','text','WriteMode','overwrite');
for i = 1:n_items
    if listing(i).isdir
        %Navigate through folders
        name_folder = listing(i).name;
        folder_listing = dir(name_folder);
        n_files = length(folder_listing);
        if(length(name_folder) > 3)
            for j = 1:n_files
                file = folder_listing(j).name;
                name_length = length(file);
                if(name_length > 3)
                    
                    file_type = file(length(file)-3:length(file));

                    if(file_type == '.wav')
                        full_file_name = join([name_folder, '/', file]);
                        %decompose the file name to get proper things for the
                        %catalogue
                        k = strfind(file,'_');
                        dotloc = strfind(file,'.');
                        ind1 = file(1:k(1)-1);
                        ind2 = file(k(1)+1:k(2)-1);
                        io = file(k(2)+1:dotloc-1);
                        % Get stats of the IR
                        try
                            [RT,DRR,CTE,CFS,EDT] =IR_stats(full_file_name);
                            new_row = {name_folder, ind1, ind2, io, RT, DRR, CTE, CFS, EDT};
                        catch
                            warning('Warning, something went wrong with IR_stats. Appending error');
                            new_row = {name_folder, ind1, ind2, io,'Error'};
                        end

                        writecell(new_row, filename,'FileType','text','WriteMode','append');
                    end
                end
            end
        end

    end
end