import unittest

from experiments.cav_guard_leave_one_domain_out import source_domains_for


class LeaveOneDomainOutTest(unittest.TestCase):
    def test_held_out_domain_is_excluded_from_guard_sources(self):
        domains = ("domain1", "domain2", "crisismmd", "mmimdb")
        for held_out in domains:
            sources = source_domains_for(domains, held_out)
            self.assertEqual(len(sources), 3)
            self.assertNotIn(held_out, sources)
            self.assertEqual(set(sources) | {held_out}, set(domains))

    def test_unknown_held_out_domain_is_rejected(self):
        with self.assertRaises(ValueError):
            source_domains_for(("domain1", "domain2"), "unknown")


if __name__ == "__main__":
    unittest.main()
