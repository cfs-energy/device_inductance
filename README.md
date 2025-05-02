# Device Inductance

[Docs](https://device-inductance.readthedocs.io/)

Tokamak core inductance matrices and flux tables.

![OSSFE2025](./docs/assets/POS-38_Logan.jpg)

See docs for detailed installation instructions and usage examples.

## Installation

```bash
pip install device_inductance
```

## Contributing

Contributions consistent with the goals and anti-goals of the package are welcome.

Please make an issue ticket to discuss changes before investing significant time into a branch.

Goals

* Library-level functions and formulas
* Comprehensive documentation including literature references, assumptions, and units-of-measure
* Quantitative unit-testing of formulas
* Performance (both speed and memory-efficiency)
    * Eliminate the need for databases to store the results of the computation
* Cross-platform compatibility
* Minimization of long-term maintenance overhead (both for the library, and for users of the library)
    * Semantic versioning
    * Automated linting and formatting tools
    * Centralized CI and toolchain configuration in as few files as possible

Anti-Goals

* Fanciness that increases environment complexity, obfuscates reasoning, or introduces platform restrictions
* Brittle CI or toolchain processes that drive increased maintenance overhead
* Application-level functionality (graphical interfaces, simulation frameworks, etc)

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
