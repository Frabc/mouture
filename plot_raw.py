import os
def makehtml(csv,html):
# import glob
# csv = glob.glob("dataset/*.csv")[0] # 1st csv in folder
#csv = "myinputfile.csv"    # Or be specific 

    import pandas as pd
    df = pd.read_csv(csv,sep=";")
    print(df)
    import plotly.graph_objects as go
    fig = go.Figure(data=[
        go.Scatter3d(   name="tip",
                        x=df['pen_tip_x'],
                        y=df['pen_tip_y'],
                        z=df['pen_tip_z'],marker={"size":1,"color":"blue"}),
        go.Scatter3d(   name="corner1",
                        x=df['corner1_x'],
                        y=df['corner1_y'],
                        z=df['corner1_z'],marker={"size":1,"color":"red"}),
        go.Scatter3d(   name="corner2",
                        x=df['corner2_x'],
                        y=df['corner2_y'],
                        z=df['corner2_z'],marker={"size":1,"color":"red"}),
        go.Scatter3d(   name="corner3",
                        x=df['corner3_x'],
                        y=df['corner3_y'],
                        z=df['corner3_z'],marker={"size":1,"color":"red"}),
    ])
    #plotfile = __file__.replace(".py",".html") # based on script name
    # plotfile = csv.replace(".csv",".html") # based on input file
    # plotfile = plotfile.replace('dataset'+os.sep, '')
    # print(f"Creating {plotfile}")
    # fig.write_html('dataset_result/'+plotfile)
    fig.write_html(html)
