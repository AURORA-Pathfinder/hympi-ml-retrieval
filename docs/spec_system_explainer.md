# Spec System Explainer
The "Spec" system is the primary means of defining data in this project.
It's based on two primary base classes: `DataSpec` and `DataSource`.

Both of which are defined in the `../data/base.py` file.

## Data Specification with `DataSpec`
`DataSpec` is a base class that defines the *specification* for a given kind of data. Consider it a means of defining what data you want and in what form.

The idea is that you'd inherit from `DataSpec` for each of your datasets like CoSMIR-H, AMPR, Nature Run, etc.

As a result of inheriting from `DataSpec`, there are a few intersting requirements and features such as:
1. All DataSpec's have optional `Filter` and `Scaler` objects. These are custom objects for doing just that defined in `../data/filter.py` and `../data/scaler.py` respectively.
2. A required `shape` property which is a tuple that defines the shape of a sample of data that would come from this spec. For AMPR, the shape would return `(8,)`.
3. A required `units` property which is a string that defines the units that the spec uses. For AMPR, this would be "Brightness Temperature (K)".
4. A required `load_raw_slice` function which is so important that it is explained below.

Note that all of the above requirements are clearly shown in the definition of `DataSpec` itself and should be referred to directly for more details. Also, there are other features for defining data-specific transformations and filtering. Refer to the example usages for CoSMIR-H as good idea of where you can take this.

#### The `load_raw_slice` Method
This method is very important and has the following signature:
```py
def load_raw_slice(self, source: DataSource, start: int, end: int) -> Sequence:
    ...
```
As a name implies, this method loads up a raw slice of data. This slice has a start and end index and uses a `DataSource` to do the slicing.

*Use of the word "raw" here refers to the data being loaded directly from the source and is not transformed or filtered in any way. The actually transforming happens on the GPU in the model training / validation / testing. It can be done manually, of course, using functions defined in `ModelDataSpec` to apply transformations to a raw batch of data following that spec.*

All DataSpec's must implement the above function to work with a specific `DataSource` or with a range of them as needed. That specific `DataSource` is then used to do the initial loading that will return a sequence of samples from the data source from start to end.

## Defining Where Data Comes from with `DataSource`
Now we know *what* data we want, now we need to know *where* we get it!

This is done with a `DataSource` another base class that is incredibly simple on purpose.

All it requires is a `sample_count` property to define how many samples of data exist in this source.

#### Recommendation: Multiple Inheritance
The best way to make use of `DataSource` is to define it for each kind of data, similar to `DataSpec`.

For AMPR, we may define a spec like `AMPRSpec` and then pair it with an `AMPRSource` (inherits from `DataSource`) that is used to define a function to start loading AMPR data. An important note is that a class that directly inherits the base `DataSource` class should not actually do any real loading, it should simply create base methods for an inheriting class to do the loading.

Consider the class `CH06Source` which is defined in `../data/ch06.py` it inherits from multiple DataSource's like `AMPRSource` and `CosmirhSource`, etc. Each of those individual sources has base functions for returning the entire est of their respective datas. Since `CH06Source` actually does loading, it has definitions for the days and file paths to load from. From there, it contains all of the functions from each of the sources it inherits to load the data from those file paths.