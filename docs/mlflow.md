## Running MLFlow
How to run MLFlow UI and open it your browser for model analysis.

1. Open terminal and navigate to project directory.
2. Type `uv run bash`
3. Navigate to the parent directory of your runs are stored (`mlruns/` directory).
4. Run `mlflow server -p <port>` where `<port>` is whatever port you'd like to host the MLFlow UI. I tend to use 8012.
5. Forward the port you have selected. This can be done in many ways, you can do it VSCode by following the instructions [here](https://code.visualstudio.com/docs/debugtest/port-forwarding).

### Other Details
For Step 3, to provide an example: my runs are located at my runs are located at `"/explore/nobackup/people/dgershm1/mlruns"` so I would navigate to `"/explore/nobackup/people/dgershm1/"` before moving on to step 4.

It may be a good idea to run the commands above in a screen session or a detachable shell that will run in the background even if you loose connection to ADAPT. Otherwise you'll need to complete the above steps everytime you login.