package xjtuse.castingcurvepredict.models;

import java.util.Vector;
import java.util.EventListener;

public class TaskManager implements ITaskEventListener {
    static Vector<TaskModel> runningTasks;
    static TaskManager mTaskManager;

    private TaskManager() {
    }

    static public TaskModel CreateTask() {
        // Write to database first

        TaskModel result = new TaskModel(123);
        return result;
    }

    static TaskModel getTaskById(int id) {
        TaskModel result = null;
        for (TaskModel iterable_element : runningTasks) {
            if (iterable_element.getId() == id) {
                result = iterable_element;
            }
        }
        return result;
    }

    static void LoadAllTrainingTasks()
    {
        
    }

    @Override
    public void onTaskStarting(TaskModel task) {
        // TODO write start status to database.

    }

    @Override
    public void onTaskStarted(TaskModel task) {
        // TODO write started status to database.

    }

    @Override
    public void onTaskStopped(TaskModel task) {
        // TODO write stopped status to database.

    }
}
