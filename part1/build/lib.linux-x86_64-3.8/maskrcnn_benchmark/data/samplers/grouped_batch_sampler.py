# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import itertools

import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.

    Arguments:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_uneven (bool): If ``True``, the sampler will drop the batches whose
            size is less than ``batch_size``

    """

    def __init__(self, sampler, group_ids, batch_size, drop_uneven=False):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids) # 每个gpu 长度7596的tensor，其中有3个值为0，剩下的为1.问题在这！应该有4个为0才对
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven

        self.groups = torch.unique(self.group_ids).sort(0)[0]   # len=2, groups=tensor([0, 1])

        self._can_reuse_batches = False

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)
        # get the sampled indices from the sampler
        sampled_ids = torch.as_tensor(list(self.sampler))
        # potentially not all elements of the dataset were sampled
        # by the sampler (e.g., DistributedSampler).
        # construct a tensor which contains -1 if the element was
        # not sampled, and a non-negative number indicating the
        # order where the element was sampled.
        # for example. if sampled_ids = [3, 1] and dataset_size = 5,
        # the order is [-1, 1, -1, 0, -1]
        order = torch.full((dataset_size,), -1, dtype=torch.int64)  # 返回dataset_size*1的，值全是-1的tensor
        order[sampled_ids] = torch.arange(len(sampled_ids)) # len7596 tensor的值有1899个正常索引值，剩下的全是-1


        # get a mask with the elements that were sampled
        mask = order >= 0


        # find the elements that belong to each individual cluster
        # 每个卡长度为2的list，包含的两个tensor的值都由True和False组成，每个长度都是7596，长度上此时没有异常！
        # 有3个clusters整个都没有True，有一个的索引1里面有7596个True。
        clusters = [(self.group_ids == i) & mask for i in self.groups]


        # get relative order of the elements inside each cluster
        # that follows the order from the sampler
        # 4个卡每个是长度为2的list，索引0是1个值的tensor，索引1是一堆值的tensor，正常长度是1898，只是未排序，下面的是排序了的
        # 有问题的索引0是空tensor，索引1长度是1899，原因同下。
        # 问题在于，这里的索引0是必须要有个值的，但上面如果group ids包含的0不够，这里就会不够。
        relative_order = [order[cluster] for cluster in clusters]

        # with the relative order, find the absolute order in the
        # sampled space
        # 每张卡长度为2，第一个是有一个值的tensor，第二个是有[0,1,...,1898]的tensor，有问题的出在某个开头的1个值tensor，是空的，它接的0-1898。
        # 索引1处的0-1898其实是排除了0处那个单独tensor的值的，因为有问题的0处值为空，所以后面是完整的0-1898，长度为1899.其余长度1898.
        permutation_ids = [s[s.sort()[1]] for s in relative_order]

        # permute each cluster so that they follow the order from
        # the sampler
        # 每张卡长度为2，第一个是有一个值的tensor，第二个是有一堆值的tensor，是query+s吗？不是。长度是3个1+1898，1个1+1899
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]


        # splits each cluster in batch_size, and merge as a list of tensors
        # split对每个卡长度为2，是一个大list套两个中list，中list里面是tensor。这里就有那个空tensor了
        splits = [c.split(self.batch_size) for c in permuted_clusters]

        # 4卡，3个长度1899，1个长度1900.1900的是卡3吗 # 1900长度的那个的第一项是一个空的tensor！
        merged = tuple(itertools.chain.from_iterable(splits))


        # now each batch internally has the right order, but
        # they are grouped by clusters. Find the permutation between
        # different batches that brings them as close as possible to
        # the order that we have in the sampler. For that, we will consider the
        # ordering as coming from the first element of each batch, and sort
        # correspondingly
        first_element_of_batch = [t[0].item() for t in merged]
        # get and inverse mapping from sampled indices and the position where
        # they occur (as returned by the sampler)
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        # from the first element in each batch, get a relative ordering
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch]
        )

        # permute the batches so that they approximately follow the order
        # from the sampler
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        # finally, permute the batches
        batches = [merged[i].tolist() for i in permutation_order]

        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)
