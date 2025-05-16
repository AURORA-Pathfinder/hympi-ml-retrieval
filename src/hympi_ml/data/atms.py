from abc import abstractmethod
import collections.abc

from hympi_ml.data.base import DataSpec, DataSource

ATMS_NEDT = [
    0.7,
    0.8,
    0.9,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.75,
    1.2,
    1.2,
    1.5,
    2.4,
    3.6,
    0.5,
    0.6,
    0.8,
    0.8,
    0.8,
    0.8,
    0.9,
]


class ATMSSource(DataSource):
    @property
    @abstractmethod
    def atms(self) -> collections.abc.Sequence:
        pass


class ATMSSpec(DataSpec):
    nedt: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        return (22,)

    @property
    def units(self) -> str:
        return "Brightness Temperature (K)"

    def load_raw_slice(
        self, source: ATMSSource, start: int, end: int
    ) -> collections.abc.Sequence:
        if not isinstance(source, ATMSSource):
            raise Exception(
                "The provided source does not have support for loading ATMS data."
            )

        return source.atms[start:end]


# def apply_nedt(atms: tf.data.Dataset) -> tf.data.Dataset:
#     """Takes atms data and applies transformations based on whether to include NEDT
#         into a new tensorflow dataset.

#     Args:
#         atms (tf.data.Dataset): A tensorflow dataset representing ATMS data.

#     Returns:
#         tf.data.Dataset: A Tensorflow dataset with the spec applied.
#     """

#     def nedt_add(atms):
#         # finds seed based on first two values of atms input (each atms input creates deterministic random noise)
#         seed = tf.cast(tf.slice(atms, [0], [2]), tf.int32)

#         nedt = tf.convert_to_tensor(ATMS_NEDT, tf.float64)
#         noise = tf.stack(
#             [
#                 tf.random.stateless_normal(
#                     shape=(), stddev=n, dtype=tf.float64, seed=seed
#                 )
#                 for n in tf.unstack(nedt)
#             ]
#         )
#         return atms + noise

#     return atms.map(map_func=nedt_add, num_parallel_calls=tf.data.AUTOTUNE)
