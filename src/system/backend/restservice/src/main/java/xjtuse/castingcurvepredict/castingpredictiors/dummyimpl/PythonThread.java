package xjtuse.castingcurvepredict.castingpredictiors.dummyimpl;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class PythonThread implements Runnable {
    private static String condaEnv = "C:\\Users\\redal\\source\\tools\\miniconda3";
    private static String activateCondaCmd = "cmd.exe \"/K\" C:\\Users\\redal\\source\\tools\\miniconda3\\Scripts\\activate.bat C:\\Users\\redal\\source\\tools\\miniconda3";
    private Thread mThread;
    private String threadName = "trainThread";

    public void start()
    {
        System.out.println("Starting new thread: " + threadName);
        if (mThread == null) {
            mThread = new Thread(this, threadName);
            mThread.start();
        }
    }

    @Override
    public void run() {
        try {
            System.out.println("开始训练程序。");
            Process process = Runtime.getRuntime().exec(activateCondaCmd
                    + "&& conda activate CastingCurvePredictEnv && python C:\\Users\\redal\\source\\repos\\CastingCurvePrediction\\src\\approaches\\rnn\\predict_app.py && exit");
            System.out.println("训练进程启动。");
            BufferedReader in = new BufferedReader(new InputStreamReader(process.getInputStream()));
            in = new BufferedReader(new InputStreamReader(process.getInputStream(), "gbk"));
            // 接收错误流
            BufferedReader isError = new BufferedReader(new InputStreamReader(process.getErrorStream(), "gbk"));
            StringBuilder sb = new StringBuilder();
            StringBuilder sbError = new StringBuilder();
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
                // sbError.append(lineError);
                // sbError.append("\n");
            }
            // System.out.println(sbError);

        } catch (Throwable t) {
            t.printStackTrace();
        }
    }

}
