package xjtuse.castingcurvepredict.castingpredictiors.dummyimpl;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class PythonThread implements Runnable {
    private static String condaEnv = "C:\\Users\\redal\\source\\tools\\miniconda3";
    private static String activateCondaCmd = "cmd.exe \"/K\" C:\\Users\\redal\\source\\tools\\miniconda3\\Scripts\\activate.bat C:\\Users\\redal\\source\\tools\\miniconda3";
    private Thread mThread = null;
    private String threadName = "trainThread";
    Process process = null;

    private long _taskId;
    private long _modelId;

    public PythonThread(long taskId, long modelId)
    {
        _taskId = taskId;
        _modelId = modelId;
    }

    public void start() {
        System.out.println("Starting new thread: " + threadName);
        if (mThread == null) {
            mThread = new Thread(this, threadName);
            mThread.start();
        }
    }

    public void stop() {
        if (process != null) {
            process.destroy();
        }

        if (mThread != null) {
            mThread.interrupt();
            mThread = null;
        }
    }

    @Override
    public void run() {
        try {
            System.out.println("开始训练程序。");
            String commandLine = activateCondaCmd
            + "&& conda activate CastingCurvePredictEnv && python C:\\Users\\redal\\source\\repos\\CastingCurvePrediction\\src\\approaches\\rnn\\predict_app.py " + _modelId + " " + _taskId + " && exit";
            process = Runtime.getRuntime().exec(commandLine);
            System.out.println("训练进程启动。");
            BufferedReader in = new BufferedReader(new InputStreamReader(process.getInputStream()));
            in = new BufferedReader(new InputStreamReader(process.getInputStream(), "gbk"));
            // 接收错误流
            BufferedReader isError = new BufferedReader(new InputStreamReader(process.getErrorStream(), "gbk"));
            StringBuilder sb = new StringBuilder();

            String line = null;
            String lineError = null;

            while ((line = in.readLine()) != null) {
                System.out.println(line);
                sb.append(line);
                sb.append("\n");
            }
            System.out.println(sb);

            while ((lineError = isError.readLine()) != null) {
                System.out.println(lineError);
            }

        } catch (Throwable t) {
            t.printStackTrace();
        }
    }

}
