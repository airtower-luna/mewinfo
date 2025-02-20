# Copyright 2025 Fiona Klute
# SPDX-License-Identifier: GPL-3.0-or-later
import argparse
import io
import json
import platform
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import timedelta
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Protocol, Self

CPUFREQ = Path('/sys/devices/system/cpu/cpufreq')
MEMINFO = Path('/proc/meminfo')
THERMAL = Path('/sys/devices/virtual/thermal')
DEVICE_TREE = Path('/proc/device-tree')
HWMON = Path('/sys/class/hwmon')


def adc_value(iio: Path, item: str) -> int | float:
    raw: int | float = int((iio / f'{item}_raw').read_text().strip())
    try:
        offset = int((iio / f'{item}_offset').read_text().strip())
        raw += offset
    except FileNotFoundError:
        pass
    try:
        scale = float((iio / f'{item}_scale').read_text().strip())
        raw *= scale
    except FileNotFoundError:
        pass
    return raw


class InfoItem(Protocol):
    @classmethod
    def parse(cls) -> Self: ...

    def json(self) -> dict[str, Any] | list[dict[str, Any]] \
        | str | int | float | bool: ...

    def __str__(self) -> str: ...


class Uname:
    @cached_property
    def _uname(self) -> platform.uname_result:
        return platform.uname()

    @classmethod
    def parse(cls) -> Self:
        return cls()

    def json(self) -> dict[str, Any]:
        return self._uname._asdict()

    def __str__(self) -> str:
        return ' '.join(s for s in self._uname if s)


@dataclass
class SystemType:
    compatible: list[str]
    model: str
    serial: str

    @classmethod
    def parse(cls) -> Self:
        return cls(
            compatible=[
                b.decode()
                for b in (DEVICE_TREE / 'compatible').read_bytes().split(b'\0')
                if b],
            model=(DEVICE_TREE / 'model').read_text().strip('\0'),
            serial=(DEVICE_TREE / 'serial-number').read_text().strip('\0'),
        )

    def json(self) -> dict[str, Any]:
        return {
            'compatible': self.compatible,
            'model': self.model,
            'serial': self.serial,
        }

    def __str__(self) -> str:
        return f'{self.model}, serial {self.serial}\n' \
            f'compatible: {', '.join(json.dumps(s) for s in self.compatible)}'


@dataclass
class CpufreqPolicy:
    cpus: set[int]
    governor: str
    stats: dict[int, int] | None
    current: int

    @classmethod
    def parse(cls, policy: Path) -> Self:
        if (policy / 'stats').is_dir():
            stats = dict()
            with (policy / 'stats/time_in_state').open() as fh:
                for line in fh:
                    f, t = line.split()
                    stats[int(f)] = int(t)
        else:
            stats = None

        return cls(
            cpus=set(
                map(int, (policy / 'affected_cpus').read_text().split())),
            governor=(policy / 'scaling_governor').read_text().strip(),
            stats=stats,
            current=int((policy / 'scaling_cur_freq').read_text()),
        )

    @cached_property
    def total_time(self) -> int | None:
        if self.stats is None:
            return None
        return sum(self.stats.values())

    def json(self) -> dict[str, Any]:
        return {
            'cpus': list(self.cpus),
            'governor': self.governor,
            'current_frequency': self.current,
            # JSON requires string keys
            'stats': dict((str(k), v) for k, v in self.stats.items())
            if self.stats else None,
        }

    def __str__(self) -> str:
        buf = io.StringIO()
        if len(self.cpus) > 1:
            cpus = str(self.cpus)
        else:
            cpus = str(next(c for c in self.cpus))
        buf.write(f'CPU {cpus} frequency governor: {self.governor}')
        if self.stats is not None:
            for freq, time in self.stats.items():
                mhz = freq // 1000
                if freq == self.current:
                    buf.write('\n* ')
                else:
                    buf.write('\n  ')
                assert self.total_time is not None
                buf.write(f'{mhz: >4} MHz  {time / self.total_time: >9.4%}')
        return buf.getvalue()


@dataclass
class Cpufreq:
    policies: list[CpufreqPolicy]

    @classmethod
    def parse(cls) -> Self:
        return cls([
            CpufreqPolicy.parse(p) for p in CPUFREQ.glob('policy[0-9]')])

    def json(self) -> list[dict[str, Any]]:
        return [p.json() for p in self.policies]

    def __str__(self) -> str:
        return '\n'.join(str(p) for p in self.policies)


@dataclass
class Meminfo:
    stats: dict[str, int]

    @classmethod
    def parse(cls) -> Self:
        data = dict()
        # see https://docs.kernel.org/filesystems/proc.html#meminfo
        with MEMINFO.open() as fh:
            for line in fh:
                if (m := re.match(r'^(.*):\s+(\d+) kB$', line)) is not None:
                    data[m.group(1)] = int(m.group(2))
        return cls(data)

    def json(self) -> dict[str, int]:
        return self.stats

    def __str__(self) -> str:
        buf = io.StringIO()
        ratio_used = 1 - self.stats['MemAvailable'] / self.stats['MemTotal']
        buf.write(f'Memory usage: {ratio_used:.2%}')
        for k in ('MemTotal', 'MemAvailable', 'Buffers', 'Cached', 'Active'):
            buf.write(f'\n{k}: {self.stats[k] // 1024} MB')
        return buf.getvalue()


class SensorKind(Enum):
    # https://www.kernel.org/doc/html/latest/hwmon/sysfs-interface.html
    VOLTAGE = ('in', 'mV')
    FAN = ('fan', None)
    FAN_PWM = ('pwm', None)
    TEMPERATURE = ('temp', 'm°C')
    CURRENT = ('curr', 'mA')
    POWER = ('power', 'µW')
    ENERGY = ('energy', 'µJ')
    HUMIDITY = ('humidity', None)
    ALARM = ('alarm', None)

    def __init__(self, prefix: str, unit: str | None):
        self.prefix = prefix
        self.unit = unit


@dataclass
class SensorValue:
    label: str | None
    kind: SensorKind
    value: int | float

    def json(self) -> dict[str, Any]:
        data = {
            'type': self.kind.name.lower(),
            'value': self.value,
        }
        if self.kind.unit:
            data['unit'] = self.kind.unit
        if self.label:
            data['label'] = self.label
        return data

    def __str__(self) -> str:
        value: int | float = self.value
        unit = self.kind.unit
        if unit is not None and unit.startswith('m'):
            value = value / 1000
            unit = unit.lstrip('m')
        return f'{self.label or self.kind.name.lower()}: {value} {unit or ''}'


class SensorNotSupported(Exception):
    pass


@dataclass
class Sensor:
    name: str
    values: list[SensorValue]
    _SPECIAL: ClassVar[dict[str, type['Sensor']]] = dict()

    @classmethod
    def _parse(cls, hwmon: Path, name: str | None = None) -> Self:
        if name is None:
            name = (hwmon / 'name').read_text().strip()
        values = list()
        for kind in SensorKind:
            for v in hwmon.glob(f'{kind.prefix}*_input'):
                s = v.name.split('_')[0]
                try:
                    label = (hwmon / f'{s}_label').read_text().strip()
                except FileNotFoundError:
                    label = None
                value = int(v.read_text())
                try:
                    offset = int((hwmon / f'{s}_offset').read_text())
                    value += offset
                except FileNotFoundError:
                    # no offset defined
                    pass
                values.append(SensorValue(label, kind, value))
        for a in hwmon.glob('*_alarm'):
            value = int(a.read_text())
            values.append(SensorValue(a.name, SensorKind.ALARM, value))
        return cls(name, values)

    @classmethod
    def parse(cls, hwmon: Path) -> 'Sensor':
        name = (hwmon / 'name').read_text().strip()
        sensor = cls._parse(hwmon, name)
        if name in cls._SPECIAL:
            try:
                return cls._SPECIAL[name].extend(hwmon, sensor)
            except SensorNotSupported:
                # special class can't handle the sensor, return
                # generic sensor data
                pass
        return sensor

    def json(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'values': [v.json() for v in self.values]
        }

    def __str__(self) -> str:
        return f'Sensor {self.name}:\n  ' \
            + '\n  '.join(str(s) for s in self.values)

    @classmethod
    def extend(cls, hwmon: Path, sensor: Self) -> Self:
        # called only for subclasses listed in _SPECIAL
        raise NotImplementedError()


class Axp20Battery(Sensor):
    """The battery controller in the Pinephone. It provides battery
    temperature, which is not available via hwmon."""
    _POWER_SUPPLY: ClassVar[str] = 'axp20x-battery-power-supply'
    _DRIVER: ClassVar[str] = 'axp20x-rsb'
    _ADC: ClassVar[str] = 'axp813-adc'

    @classmethod
    def extend(cls, hwmon: Path, base: Sensor) -> Self:
        hwmon = hwmon.resolve()
        battery = None
        for b in hwmon.parents:
            if b.name == cls._POWER_SUPPLY:
                battery = b
        if battery is None \
                or (battery.parent / 'driver').resolve().name != cls._DRIVER:
            raise SensorNotSupported(cls.name, base.name)
        adc = battery.parent / cls._ADC
        try:
            iio = next(adc.glob('iio:device*'))
        except StopIteration:
            raise SensorNotSupported(cls.name, base.name)
        base.values.append(SensorValue(
            None, SensorKind.TEMPERATURE, adc_value(iio, 'in_temp')))
        return cls(base.name, base.values)


class RpiVoltage(Sensor):
    """Low voltage alarm in Raspberry Pi boards."""

    @classmethod
    def extend(cls, hwmon: Path, base: Sensor) -> Self:
        for i, a in enumerate(base.values):
            if a.label == 'in0_lcrit_alarm':
                break
        alarm = replace(a, label='low voltage alarm')
        base.values.remove(a)
        base.values.append(alarm)
        return cls(base.name, base.values)


Sensor._SPECIAL['axp20x_battery'] = Axp20Battery
Sensor._SPECIAL['rpi_volt'] = RpiVoltage


@dataclass
class Hwmon:
    sensors: list[Sensor]

    @classmethod
    def parse(cls) -> Self:
        sensors = [Sensor.parse(h) for h in HWMON.iterdir()]
        return cls(sensors)

    def json(self) -> list[dict[str, Any]]:
        return [s.json() for s in self.sensors]

    def __str__(self) -> str:
        return '\n'.join(str(s) for s in self.sensors)


@dataclass
class Uptime:
    uptime: timedelta

    @classmethod
    def parse(cls) -> Self:
        return cls(timedelta(
            seconds=float(Path('/proc/uptime').read_text().split()[0])))

    def json(self) -> float:
        return self.uptime.total_seconds()

    def __str__(self) -> str:
        return f'up {self.uptime}'


ITEMS: dict[str, type[InfoItem]] = {
    'uname': Uname,
    'system': SystemType,
    'uptime': Uptime,
    'cpufreq': Cpufreq,
    'memory': Meminfo,
    'hwmon': Hwmon,
}


class MewArgs(argparse.Namespace):
    json: bool
    func: Callable[[Self], None]


def sensors(args: MewArgs) -> None:
    data = Hwmon.parse()

    if args.json:
        json.dump(data, sys.stdout, indent=2, default=lambda x: x.json())
        print()
    else:
        print(data)
        print()
        print('=^.^=')


def _all(args: MewArgs) -> None:
    data: dict[str, InfoItem] = dict()
    for name, cls in ITEMS.items():
        try:
            data[name] = cls.parse()
        except FileNotFoundError:
            pass

    if args.json:
        json.dump(data, sys.stdout, indent=2, default=lambda x: x.json())
        print()
    else:
        for item in data.values():
            print(item)
            print()
        print('=^.^=')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-j', '--json', action='store_true', help='output data as JSON')
    parser.set_defaults(func=_all)
    subparsers = parser.add_subparsers()
    sensors_parser = subparsers.add_parser('sensors')
    sensors_parser.set_defaults(func=sensors)
    args = parser.parse_args(namespace=MewArgs())
    args.func(args)


if __name__ == '__main__':
    main()
