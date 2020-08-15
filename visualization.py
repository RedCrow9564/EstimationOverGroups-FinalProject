import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from os.path import join, exists, dirname
from os import makedirs, chdir


def create_directory(output_path):
    dirName = dirname(output_path)

    if not exists(dirName):
        makedirs(dirName)
        print("Directory ", dirName,  " Created ")
    else:
        print("Directory ", dirName,  " already exists")


### reproducing graph 8

def reproduce_fig_8(file_path, output_path, columns_to_group_by, shorten_dist=True):

    """
    The function reads a path_file of data formatted for figure 8 in the paper.

    A plot is created according to the format of fig 8 (separate values by columns_to_group_by, plot error as function of N). The
    path defined is created including intermediate directories and the image is saved to the output_path.

    If shorten_dist ==True, then a shorter notation is used for the parameter 'Shifts Distribution'

    Args:
        file_path(Union[str, tuple]): path or paths of data to read.

        output_path(str): path of image to create.

        columns_to_group_by(tuple): by which columns to group the data.

        shorten_dist(bool): whether to use shorten notation for Shifts Distribution column. This is because it takes up
        a lot of text in graph label, and runs over graph. Default value is True.


    Returns:
        No variable. Saves graph in output_path.
    """

    create_directory(output_path)

    relevant_data = pd.DataFrame()

    if type(file_path) == str:
        relevant_data = pd.read_csv(file_path, header=0)

    else:
        for curr_path in file_path:
            curr_rel_data = pd.read_csv(curr_path, header=0)
            relevant_data = relevant_data.append(curr_rel_data)



    relevant_data_grouped = relevant_data.groupby(list(columns_to_group_by) + ['Observations Number'],
                                                  as_index=False)['Mean Error'].mean()

    if ('Shifts Distribution' in columns_to_group_by) and shorten_dist:
        relevant_data_grouped.loc[:, r'Shifts Distribution'] = relevant_data_grouped[r'Shifts Distribution'].apply(lambda x: x.split(' ')[0])
        relevant_data_grouped = relevant_data_grouped.rename(columns={r'Shifts Distribution': 'Distribution'})
        columns_to_group_by = [col for col in columns_to_group_by if col != r'Shifts Distribution']+['Distribution']

    fig, ax = plt.subplots()

    for name, group in relevant_data_grouped.groupby(list(columns_to_group_by)):
        print(name)

        label = ''

        for ii, col in enumerate(columns_to_group_by):
            if col == "Noise power":
                label += r'$\sigma^{2}=$' + str(name[ii])
            elif col == "Distribution":
                label += '\\textit{' + str(name[ii]) + ' Distribution}'
            else:
                label += '\\textit{' + col + '}' + ' = ' + str(name[ii])
            if ii < len(columns_to_group_by)-1:
                label += ', '

        group.plot(x='Observations Number', y='Mean Error', ax=ax, label=label, loglog=True)

    plt.legend(loc='best', fontsize=10)
    ax.set_ylabel('\\textit{Mean Error}', fontsize=12)
    ax.set_xlabel('\\textit{Observations Number}', fontsize=12)
    plt.title('\\textit{MRFA Estimation Error against the observations number for} $L=20$')
    plt.savefig(output_path)
    # plt.show()


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"})


path = r'C:\Users\Elad\Google Drive\MScStudies\Courses\Estimation and Approximation Problems over Groups'
path += r'\FinalProject\Results\Experiment Results'
chdir(path)

### single file
# path_input = r'complex L=20 no noise.csv'

### more than one file
path_input = (r'complex L=20 no noise.csv', r'complex L=20 with noise results.csv',
              r'complex L=20 no noise-Dirac distribution results.csv',
              r'complex L=20 with noise-Dirac distribution results.csv')


output_path = join(r'output', 'fig_7-complex_case.png')

### which columns to use for the graph
columns_to_group_by = ('r', 'Noise power', 'Shifts Distribution')

### whether to use shorten notation for distribution
shorten_dist = True


#reproduce_fig_8(file_path=path_input, output_path=output_path, columns_to_group_by=columns_to_group_by, shorten_dist=shorten_dist)





def reproduce_fig_7(file_path, output_path, datatype = 'complex128', dist = 'Uniform Distribution'):
    """
    The function reads a path_file of data formatted for figure 7 in the paper.

    A plot is created according to the format of fig 7 (heatmap by r and L, log10 maximal_error).  The
    output path defined is created including intermediate directories and the image is saved to the output_path.

    Args:

        file_path(str): path of data to read.

        output_path(str): path of image to create.


    Returns:
        No variable. Saves graph in output_path.
    """

    create_directory(output_path)


    relevant_data = pd.read_csv(file_path, header=0)

    relevant_data = relevant_data[(relevant_data['Shifts Distribution'] ==dist)&(relevant_data['Data type (complex/real)'] ==datatype)]

    relevant_data.loc[:, 'log_max_error'] = np.log10(relevant_data['Max Error']).round(decimals = 1)

    relevant_data_pivot = relevant_data.pivot(index='r', columns='Data size', values='log_max_error')

    relevant_data_pivot = relevant_data_pivot.sort_index(ascending=False)

    fig, ax = plt.subplots()

    sns.heatmap(relevant_data_pivot, annot=True, ax=ax, cmap='Blues')

    ax.set_ylabel('\\textit{r}', fontsize=16)
    ax.set_xlabel('\\textit{L}', fontsize=16)
    plt.title(r'$\log_{10}$ \textit{of Maximal Error for Different L and r - Real case}', fontsize=12)

    plt.savefig(output_path)

    return




path = r'C:\Users\imenu\Desktop\studies\estimation_groups\project\visualization'
#chdir(path)

### single file
path_input = r'C:\Users\Elad\Google Drive\MScStudies\Courses\Estimation and Approximation Problems over Groups'
path_input += r'\FinalProject\Results\Experiment Results\Fig 7 real\Real fig 7 results.csv'

### more than one file
# path_input = (r'complex L=20 no noise.csv', r'complex L=20 with noise results.csv',
#               r'complex L=20 no noise-Dirac distribution results.csv',
#               r'complex L=20 with noise-Dirac distribution results.csv')


output_path = join(r'output', 'fig_7-real_case.png')


### these values are already defined as default values, but added them in still so it would be intuitive to replace
datatype = 'float64'
dist = 'Uniform Distribution'


reproduce_fig_7(file_path=path_input, output_path=output_path, datatype = datatype, dist = dist)