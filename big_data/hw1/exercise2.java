import java.io.*;
import java.util.*;
import java.lang.Math;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapred.lib.MultipleInputs;

public class exercise2 extends Configured implements Tool {
    public static class Map1 extends MapReduceBase implements Mapper<LongWritable, Text, Text, Text> {
        private Text keyword = new Text("std");

        public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
            String line = value.toString();
            String[] words = line.split("\\s+");

            Double volume = Double.parseDouble(words[3]);
            Double volumeSquare = volume*volume;
            
            String count = "1";

            String outputValue = volume + "," + volumeSquare + "," + count;

            output.collect(keyword, new Text(outputValue));         
        }
    }

    public static class Map2 extends MapReduceBase implements Mapper<LongWritable, Text, Text, Text> {
        private Text keyword = new Text("std");

        public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
            String line = value.toString();
            String[] words = line.split("\\s+");

            Double volume = Double.parseDouble(words[4]);
            Double volumeSquare = volume*volume;
            
            String count = "1";

            String outputValue = volume + "," + volumeSquare + "," + count;

            output.collect(keyword, new Text(outputValue));
        }
    }

    public static class Combiner extends MapReduceBase implements Reducer<Text, Text, Text, Text> {
        public void reduce (Text key, Iterator<Text> values, OutputCollector<Text, Text> context, Reporter reporter) throws IOException{
            Double sum = 0.0;
            Double sumSquare = 0.0;
            Double count = 0.0;

            while (values.hasNext()) {
                String valueStr = values.next().toString();
                String[] elements = valueStr.split(",");
                sum += Double.parseDouble(elements[0]);
                sumSquare += Double.parseDouble(elements[1]);
                count += Double.parseDouble(elements[2]);
            }

            String outputValue = sum + "," + sumSquare + "," + count;

            context.collect(key, new Text(outputValue));
        }
    }

    public static class Reduce extends MapReduceBase implements Reducer<Text, Text, Text, DoubleWritable> {
        public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, DoubleWritable> output, Reporter reporter) throws IOException {
            Double sum = 0.0;
            Double sumSquare = 0.0;
            Double count = 0.0;

            while (values.hasNext()) {
                String valueStr = values.next().toString();
                String[] elements = valueStr.split(",");
                sum += Double.parseDouble(elements[0]);
                sumSquare += Double.parseDouble(elements[1]);
                count += Double.parseDouble(elements[2]);
            }

            Double std = Math.sqrt(sumSquare/count - (sum/count)*(sum/count));

            output.collect(key, new DoubleWritable(std));
        }
    }

    
    public int run(String[] args) throws Exception {
        JobConf conf = new JobConf(getConf(), exercise2.class);
		conf.setJobName("hw1_exercise2_lf");

        conf.setMapOutputKeyClass(Text.class);
	    conf.setMapOutputValueClass(Text.class);
		conf.setOutputKeyClass(Text.class); 
		conf.setOutputValueClass(Text.class);

		conf.setMapperClass(Map1.class);
		conf.setMapperClass(Map2.class);
        conf.setCombinerClass(Combiner.class);
		conf.setReducerClass(Reduce.class);

		conf.setInputFormat(TextInputFormat.class);
		conf.setOutputFormat(TextOutputFormat.class);

	    MultipleInputs.addInputPath(conf, new Path(args[0]), TextInputFormat.class,Map1.class);
	    MultipleInputs.addInputPath(conf, new Path(args[1]), TextInputFormat.class,Map2.class);
	    FileOutputFormat.setOutputPath(conf, new Path(args[2]));

		JobClient.runJob(conf);
		return 0;
    }
    
    public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new exercise2(), args);
		System.exit(res);
    }
}

