package xjtuse.castingcurvepredict.viewmodels;

import java.util.ArrayList;
import java.util.List;

public class TaskCollectionViewModel {
    private List<TaskViewModel> taskViewModels = new ArrayList<TaskViewModel>();

    public TaskCollectionViewModel(List<TaskViewModel> viewModels)
    {
        taskViewModels = viewModels;
    }

    public List<TaskViewModel> getTaskViewModels()
    {
        return taskViewModels;
    }
}
