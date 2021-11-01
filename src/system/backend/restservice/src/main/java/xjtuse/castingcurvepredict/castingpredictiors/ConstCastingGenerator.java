package xjtuse.castingcurvepredict.castingpredictiors;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.Buffer;
import java.util.Scanner;

import xjtuse.castingcurvepredict.interfaces.*;
import xjtuse.castingcurvepredict.models.CastingInputModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;

public class ConstCastingGenerator implements ICastingGenerator {
    public ConstCastingGenerator()
    {
        
    }

    @Override
    public CastingResultModel PredcitCastingCurve(CastingInputModel input) {
        // Read data from json
        CastingResultModel resultModel = new CastingResultModel();

        BufferedReader bfReader;
        try {
            bfReader = new BufferedReader(new FileReader("C:/Users/redal/source/repos/CastingCurvePrediction/datasets/data.csv"));
            Boolean firstLine = true;
            String line;
            while((line = bfReader.readLine()) != null)
            {
                if(firstLine)
                {
                    firstLine = false;
                }
                else{
                    String itemToSplit = line;
                    String[] item = itemToSplit.split(",");
                    String date = item[0];
                    double value = Double.parseDouble(item[1]);
                    resultModel.addResultItem(date, value);
                }
            }
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return resultModel;
    }

}
