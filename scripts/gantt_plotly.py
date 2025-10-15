# import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff
import argparse
 
parser = argparse.ArgumentParser(description='Plot Gantt chart')
parser.add_argument('-i', '--input_path', type=str, help='Input file path', default='timeline.csv', required=False)
parser.add_argument('-o', '--output_path', type=str, help='Output file path', default='timeline.html', required=False)

def return_type(x):
    if x[0:2] == 'DI':
        return 'DMA_IN'
    elif x[0:2] == 'DO':
        return 'DMA_OUT'
    else:
        return 'Slice'

def return_colors(operators):
    import random
    colors = {}
    used_colors = {'': 0}
    for x in operators:
        if x in colors:
            continue

        color = ''
        while color in used_colors:
            color = 'rgb({}, {}, {})'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors[x] = color
    return colors

def plot(input_path, output_path):
    source = pd.read_csv(input_path, names=['Task', 'Start', 'Finish', 'LineNo', 'Operator'])
    colors = return_colors(source['Operator'])
    source['TaskType'] = source.apply(lambda x: return_type(x['Task']), axis=1)
    source['Colors'] = source.apply(lambda x: colors[x['Operator']], axis=1)

    source.sort_values(by=['Start'], inplace=True)

    print(source)
    print(colors)

    fig = ff.create_gantt(source, index_col = 'Operator',  bar_width = 0.4, colors=colors)
    fig.update_layout(xaxis_type='linear', autosize=False, width=2000, height=1500)

    fig.write_html(output_path)

if __name__ == '__main__':
    args = parser.parse_args()
    plot(args.input_path, args.output_path)