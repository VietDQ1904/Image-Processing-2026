import torch

class Logger(object):
    def __init__(self, runs, info=None, log_path=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.log_path = log_path

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def _log(self, text):
        print(text)
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()

            self._log(f'Run {run + 1:02d}:')
            self._log(f'Highest Train: {result[:, 0].max():.2f}')
            self._log(f'Highest Valid: {result[:, 1].max():.2f}')
            self._log(f'  Final Train: {result[argmax, 0]:.2f}')
            self._log(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            self._log('All runs:')
            r = best_result[:, 0]
            self._log(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 1]
            self._log(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 2]
            self._log(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 3]
            self._log(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
