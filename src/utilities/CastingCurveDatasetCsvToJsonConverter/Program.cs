using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using CsvHelper;

namespace CastingCurveDatasetCsvToJsonConverter
{
    class Program
    {
        private static string _datasetFolder = "../../../datasets/";
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            List<CastingCurveCsvItem> datasets = null;
            using (var reader = new StreamReader(_datasetFolder + "data2.csv"))
            using (var csv = new CsvReader(reader, System.Globalization.CultureInfo.InvariantCulture))
            {
                var records = csv.GetRecords<CastingCurveCsvItem>();
                datasets = records.ToList();
            }
            List<CastingCurveJsonItem> jsonItems = new List<CastingCurveJsonItem>();
            CastingCurveJsonModel jsonModel = new CastingCurveJsonModel();
            List<string> times = new List<string>();
            List<double> values = new List<double>();
            for (int i = 0; i < datasets.Count; ++i)
            {
                jsonItems.Add(
                new CastingCurveJsonItem
                {
                    no = i + 1,
                    datetime = datasets[i].ds,
                    value = datasets[i].y
                });
                times.Add(datasets[i].ds);
                values.Add(datasets[i].y);
            }
            jsonModel.times = times.ToArray();
            jsonModel.values = values.ToArray();
            var jsonString = JsonConvert.SerializeObject(new {
                CastingCurveValues = jsonModel
            });

            File.WriteAllText(_datasetFolder + "data2.json", jsonString);
        }
    }
}
